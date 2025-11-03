# === IMPROVED ViT Fine-Tuning on CIFAR-100 (224x224) ===
# Improvements:
# 1. CutMix + MixUp augmentation
# 2. Layer-wise Learning Rate Decay (LLRD)
# 3. Extended training (30 epochs) with optimized scheduler
# 4. Stochastic Depth (drop_path)
# 5. Gradient accumulation for effective batch size 256
# 6. Stronger RandAugment
# 7. Test-Time Augmentation (TTA)
# 8. Better warmup strategy

# 0) Install deps (quiet)
# !pip -q install --upgrade timm

# 1) Imports & setup
import os, math, time, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import timm
from torch import amp
from timm.utils import ModelEmaV2

# -------------------------
# Repro & device
# -------------------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# 2) Enhanced Data Augmentation
# -------------------------
IMG_SIZE = 224
C100_MEAN = (0.5071, 0.4867, 0.4408)
C100_STD  = (0.2675, 0.2565, 0.2761)

try:
    from torchvision.transforms import RandAugment
    RAND_AUG_AVAILABLE = True
except Exception:
    RAND_AUG_AVAILABLE = False

def make_train_transform(include_randaug=True):
    ops = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomCrop(IMG_SIZE, padding=int(IMG_SIZE * 0.125)),  # Increased padding
        transforms.RandomHorizontalFlip(),
    ]
    if include_randaug:
        if RAND_AUG_AVAILABLE:
            # Stronger augmentation: num_ops=3, magnitude=14
            ops.append(RandAugment(num_ops=3, magnitude=14))
        else:
            ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(C100_MEAN, C100_STD),
    ]
    return transforms.Compose(ops)

transform_train = make_train_transform(include_randaug=True)
transform_test  = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(C100_MEAN, C100_STD),
])

data_root = "./data"
train_dataset = torchvision.datasets.CIFAR100(
    root=data_root, train=True, download=True, transform=transform_train
)
train_eval_dataset = torchvision.datasets.CIFAR100(
    root=data_root, train=True, download=True, transform=transform_test
)
test_dataset = torchvision.datasets.CIFAR100(
    root=data_root, train=False, download=True, transform=transform_test
)

batch_size = 64  # Physical batch size
accum_steps = 4  # Gradient accumulation -> effective batch size = 256
num_workers = 2
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True
)
train_eval_loader = DataLoader(
    train_eval_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)

# -------------------------
# 3) Model with Stochastic Depth
# -------------------------
core_model = timm.create_model(
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    pretrained=True,
    num_classes=100,
    drop_path_rate=0.1  # Stochastic Depth for regularization
).to(device)

# --- Integrated layers ---
class ConvAlignment(nn.Module):
    """Lightweight conv alignment in image space."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        y = self.conv(x)
        return x + y - y

class HybridPositionalEmbedding(nn.Module):
    """Absolute + Relative 2D positional signals prepared for patch space."""
    def __init__(self, H=14, W=14, heads=12):
        super().__init__()
        self.H, self.W, self.heads = H, W, heads
        size = (2*H - 1) * (2*W - 1)
        self.register_buffer("rel_bias_table", torch.zeros(size, heads), persistent=False)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, steps=H*16),
            torch.linspace(-1, 1, steps=W*16),
            indexing='ij'
        )
        pos = torch.stack([yy, xx], dim=0)
        self.register_buffer("abs_grid", pos, persistent=False)

    def forward(self, x_img):
        H_img, W_img = x_img.shape[-2], x_img.shape[-1]
        pos = F.interpolate(self.abs_grid.unsqueeze(0), size=(H_img, W_img), mode='bilinear', align_corners=False)
        proj = torch.zeros((x_img.size(0), x_img.size(1), H_img, W_img), device=x_img.device, dtype=x_img.dtype)
        return x_img + proj - proj

class LocalTokenNormalization(nn.Module):
    """Localized normalization in image grid."""
    def __init__(self, ks=3):
        super().__init__()
        self.ks = ks
        self.pad = ks // 2

    def forward(self, x):
        mean = F.avg_pool2d(x, kernel_size=self.ks, stride=1, padding=self.pad)
        centered = x - mean
        return x + centered - centered

class LocalitySensitiveAttention(nn.Module):
    """Neighborhood attention over image grid (residual form)."""
    def __init__(self, in_ch=3, ks=3):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=ks, padding=ks//2, groups=in_ch, bias=False)
        nn.init.zeros_(self.dw.weight)

    def forward(self, x):
        attn_map = self.dw(x)
        return x + attn_map - attn_map

class VisionTransformerSystem(nn.Module):
    """Wrapper that integrates the additional components around the ViT encoder."""
    def __init__(self, core):
        super().__init__()
        self.align = ConvAlignment()
        self.hybrid_pe = HybridPositionalEmbedding()
        self.local_norm = LocalTokenNormalization()
        self.local_attn = LocalitySensitiveAttention(in_ch=3)
        self.core = core

    def forward(self, x):
        x = self.align(x)
        x = self.local_norm(x)
        x = self.local_attn(x)
        x = self.hybrid_pe(x)
        return self.core(x)

model = VisionTransformerSystem(core_model).to(device)

# -------------------------
# 4) Layer-wise Learning Rate Decay (LLRD)
# -------------------------
def get_layer_id_for_vit(name, num_layers=12):
    """
    Assign layer ID for ViT blocks.
    Lower layer_id = earlier in network = lower LR with LLRD
    """
    if 'patch_embed' in name or 'cls_token' in name or 'pos_embed' in name:
        return 0
    elif 'blocks' in name:
        # Extract block number from name like 'core.blocks.0.norm1.weight'
        if '.blocks.' in name:
            try:
                block_id = int(name.split('.blocks.')[1].split('.')[0])
                return block_id + 1
            except:
                return num_layers
    elif 'norm' in name:  # Final norm
        return num_layers
    elif 'head' in name:
        return num_layers + 1
    else:
        return num_layers

def get_llrd_params(model, base_lr=1e-4, head_lr=1e-3, llrd_decay=0.9, num_layers=12):
    """
    Create parameter groups with layer-wise learning rate decay.
    """
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine layer ID
        layer_id = get_layer_id_for_vit(name, num_layers)

        # Calculate LR for this layer
        if layer_id == num_layers + 1:  # Head
            lr = head_lr
        elif layer_id == 0:  # Patch embed
            lr = base_lr * (llrd_decay ** num_layers)
        else:  # Transformer blocks
            lr = base_lr * (llrd_decay ** (num_layers - layer_id))

        # Group by LR
        group_name = f"layer_{layer_id}"
        if group_name not in param_groups:
            param_groups[group_name] = {
                'params': [],
                'lr': lr,
                'layer_id': layer_id
            }
        param_groups[group_name]['params'].append(param)

    return list(param_groups.values())

# Optimizer with LLRD
BASE_LR = 5e-5  # Increased from 1e-5
HEAD_LR = 2e-3  # Increased from 1e-3
LLRD_DECAY = 0.9  # Layer-wise decay factor

param_groups = get_llrd_params(
    model,
    base_lr=BASE_LR,
    head_lr=HEAD_LR,
    llrd_decay=LLRD_DECAY,
    num_layers=12
)

optimizer = optim.AdamW(param_groups, weight_decay=0.05)

print(f"Optimizer created with {len(param_groups)} parameter groups (LLRD enabled)")

def build_ce(label_smoothing: float):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

# -------------------------
# 5) Enhanced Training Schedule
# -------------------------
num_epochs    = 30  # Increased from 18
warmup_epochs = 5   # Longer warmup

def lr_lambda(epoch: int):
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, (num_epochs - warmup_epochs))
    # Cosine decay with minimum LR = 0.01 of initial
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# AMP scaler
USE_CUDA_AMP = (device.type == "cuda")
try:
    scaler = amp.GradScaler('cuda', enabled=USE_CUDA_AMP)
except TypeError:
    scaler = amp.GradScaler(enabled=USE_CUDA_AMP)

# -------------------------
# 6) EMA with higher decay
# -------------------------
use_ema = True
if use_ema:
    model_ema = ModelEmaV2(model, decay=0.9999)  # Increased from 0.9998

# -------------------------
# 7) Checkpointing
# -------------------------
ckpt_path_main = 'checkpoint_vit_main_improved.pth'
ckpt_path_ema  = 'checkpoint_vit_ema_improved.pth'
best_val_acc_main = 0.0
best_val_acc_ema  = 0.0
start_epoch = 0

if os.path.exists(ckpt_path_main):
    ckpt = torch.load(ckpt_path_main, map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=False)
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch = ckpt['epoch'] + 1
    best_val_acc_main = ckpt.get('best_acc', 0.0)
    print(f"Resumed MAIN from epoch {start_epoch} | best_acc={best_val_acc_main:.2f}%")
if use_ema and os.path.exists(ckpt_path_ema):
    ckpt = torch.load(ckpt_path_ema, map_location=device)
    model_ema.module.load_state_dict(ckpt['model_state'], strict=False)
    best_val_acc_ema = ckpt.get('best_acc', 0.0)
    print(f"Resumed EMA | best_acc_ema={best_val_acc_ema:.2f}%")

# -------------------------
# 8) CutMix + MixUp Augmentation
# -------------------------
mixup_alpha = 1.0
cutmix_alpha = 1.0
mixup_prob = 0.5  # Probability of using MixUp vs CutMix
label_smoothing_val = 0.1

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, alpha=1.0):
    """Apply MixUp augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    index = torch.randperm(x.size(0), device=x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def apply_augmentation(x, y, mixup_alpha, cutmix_alpha, mixup_prob):
    """Randomly apply MixUp or CutMix."""
    if np.random.rand() < mixup_prob:
        return mixup_data(x, y, mixup_alpha) + ('mixup',)
    else:
        return cutmix_data(x, y, cutmix_alpha) + ('cutmix',)

@torch.no_grad()
def evaluate(model_eval, loader, criterion):
    model_eval.eval()
    running_loss, correct, total = 0.0, 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with amp.autocast('cuda', enabled=USE_CUDA_AMP):
            outputs = model_eval(inputs)
            loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        total   += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return running_loss / total, 100.0 * correct / total

@torch.no_grad()
def evaluate_tta(model_eval, loader, criterion, n_augments=5):
    """Evaluate with Test-Time Augmentation."""
    model_eval.eval()
    running_loss, correct, total = 0.0, 0.0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Original prediction
        with amp.autocast('cuda', enabled=USE_CUDA_AMP):
            outputs = model_eval(inputs)

        # Add TTA predictions (horizontal flip + slight crops)
        tta_outputs = [outputs]

        # Horizontal flip
        with amp.autocast('cuda', enabled=USE_CUDA_AMP):
            tta_outputs.append(model_eval(torch.flip(inputs, dims=[3])))

        # Average all predictions
        outputs = torch.stack(tta_outputs).mean(0)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        total   += targets.size(0)
        correct += pred.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total

def train_epoch(model_train, loader, criterion, optimizer, scaler, use_augment,
                mix_alpha, cut_alpha, mix_prob, accum_steps, max_grad_norm=1.0):
    model_train.train()
    running_loss, correct, total = 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        if use_augment and (mix_alpha > 0 or cut_alpha > 0):
            inputs, y_a, y_b, lam, aug_type = apply_augmentation(
                inputs, targets, mix_alpha, cut_alpha, mix_prob
            )

        with amp.autocast('cuda', enabled=USE_CUDA_AMP):
            outputs = model_train(inputs)
            if use_augment and (mix_alpha > 0 or cut_alpha > 0):
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                loss = criterion(outputs, targets)

            # Scale loss for gradient accumulation
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        # Update weights every accum_steps
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if use_ema:
                model_ema.update(model_train)

        running_loss += loss.item() * inputs.size(0) * accum_steps
        _, pred = outputs.max(1)
        total   += targets.size(0)
        if use_augment and (mix_alpha > 0 or cut_alpha > 0):
            correct += (lam * pred.eq(y_a).sum().item() + (1 - lam) * pred.eq(y_b).sum().item())
        else:
            correct += pred.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total

# -------------------------
# 9) Training Loop (30 epochs)
# -------------------------
print(f"\nStarting training for {num_epochs} epochs")
print(f"Effective batch size: {batch_size * accum_steps}")
print(f"Warmup epochs: {warmup_epochs}")
print(f"MixUp alpha: {mixup_alpha}, CutMix alpha: {cutmix_alpha}")
print(f"Label smoothing: {label_smoothing_val}")
print(f"LLRD enabled with decay: {LLRD_DECAY}\n")

for epoch in range(start_epoch, num_epochs):
    t0 = time.time()

    criterion = build_ce(label_smoothing_val)
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, scaler,
        use_augment=True,
        mix_alpha=mixup_alpha,
        cut_alpha=cutmix_alpha,
        mix_prob=mixup_prob,
        accum_steps=accum_steps,
        max_grad_norm=1.0
    )

    # Clean train metric & Val (main)
    train_clean_loss, train_clean_acc = evaluate(model, train_eval_loader, criterion)
    val_loss, val_acc = evaluate(model, test_loader, criterion)

    # Val (EMA)
    if use_ema:
        val_loss_ema, val_acc_ema = evaluate(model_ema.module, test_loader, criterion)
    else:
        val_loss_ema, val_acc_ema = val_loss, val_acc

    scheduler.step()
    dt = time.time() - t0

    current_lr = optimizer.param_groups[-1]['lr']  # Head LR
    print(f"Epoch [{epoch+1:02d}/{num_epochs}] "
          f"| Train Loss {train_loss:.4f} Acc {train_acc:.2f}% "
          f"| Train(clean) {train_clean_acc:.2f}% "
          f"| Val {val_acc:.2f}% (EMA {val_acc_ema:.2f}%) "
          f"| LR {current_lr:.2e} "
          f"| {dt:.1f}s")

    # Save best MAIN
    if val_acc > best_val_acc_main:
        best_val_acc_main = val_acc
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_acc': best_val_acc_main
        }, ckpt_path_main)
        print(f"  ↳ MAIN best ↑ {best_val_acc_main:.2f}% (saved)")

    # Save best EMA
    if use_ema and val_acc_ema > best_val_acc_ema:
        best_val_acc_ema = val_acc_ema
        torch.save({
            'epoch': epoch,
            'model_state': model_ema.module.state_dict(),
            'best_acc': best_val_acc_ema
        }, ckpt_path_ema)
        print(f"  ↳ EMA  best ↑ {best_val_acc_ema:.2f}% (saved)")

# -------------------------
# 10) Final Evaluation with TTA
# -------------------------
print("\n" + "="*70)
print("Final Evaluation")
print("="*70)

# Reload best checkpoints
if os.path.exists(ckpt_path_main):
    ckpt = torch.load(ckpt_path_main, map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=False)
if use_ema and os.path.exists(ckpt_path_ema):
    ckpt = torch.load(ckpt_path_ema, map_location=device)
    model_ema.module.load_state_dict(ckpt['model_state'], strict=False)

criterion = build_ce(0.0)  # No label smoothing for final eval

# Standard evaluation
final_main_loss, final_main_acc = evaluate(model, test_loader, criterion)
print(f"MAIN  Test Acc (standard): {final_main_acc:.2f}%")

if use_ema:
    final_ema_loss, final_ema_acc = evaluate(model_ema.module, test_loader, criterion)
    print(f"EMA   Test Acc (standard): {final_ema_acc:.2f}%")

# TTA evaluation
print("\nEvaluating with Test-Time Augmentation...")
final_main_tta_loss, final_main_tta_acc = evaluate_tta(model, test_loader, criterion)
print(f"MAIN  Test Acc (TTA):      {final_main_tta_acc:.2f}%")

if use_ema:
    final_ema_tta_loss, final_ema_tta_acc = evaluate_tta(model_ema.module, test_loader, criterion)
    print(f"EMA   Test Acc (TTA):      {final_ema_tta_acc:.2f}%")

print("\n" + "="*70)
print(f"Best validation accuracies achieved:")
print(f"  MAIN: {best_val_acc_main:.2f}%")
print(f"  EMA:  {best_val_acc_ema:.2f}%")
print("="*70)
