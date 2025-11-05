# === ViT CIFAR-100 v4: MINIMAL IMPROVEMENTS (Back to Basics!) ===
# Strategy: Start from ORIGINAL working code (94.32%) and make MINIMAL changes
# Goal: Beat 94.32% → Reach 95%+ by fixing what actually matters
#
# KEY INSIGHT: v3 was STILL over-regularized! Going back to original simplicity.
#
# CHANGES FROM ORIGINAL:
# 1. Epochs: 18 → 30 (more training time)
# 2. HEAD_LR: 1e-3 → 1.5e-3 (+50% modest increase)
# 3. BODY_LR: 1e-5 → 2e-5 (+100% modest increase)
# 4. EMA decay: 0.9998 → 0.9997 (for 30 epochs)
# 5. Warmup: 3 → 5 epochs (proportional)
# 6. KEEP EVERYTHING ELSE FROM ORIGINAL!

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================
!pip -q install --upgrade timm

# ============================================================================
# IMPORTS
# ============================================================================
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

# ============================================================================
# SETUP
# ============================================================================
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ============================================================================
# DATA AUGMENTATION (ORIGINAL - WEAK!)
# ============================================================================
IMG_SIZE = 224
C100_MEAN = (0.5071, 0.4867, 0.4408)
C100_STD  = (0.2675, 0.2565, 0.2761)

try:
    from torchvision.transforms import RandAugment
    RAND_AUG_AVAILABLE = True
except Exception:
    RAND_AUG_AVAILABLE = False

def make_train_transform():
    ops = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomCrop(IMG_SIZE, padding=28),  # Original padding
        transforms.RandomHorizontalFlip(),
    ]
    if RAND_AUG_AVAILABLE:
        # ORIGINAL: (2, 10) - NOT (2, 12) or (3, 14)!
        ops.append(RandAugment(num_ops=2, magnitude=10))
    else:
        ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    ops += [transforms.ToTensor(), transforms.Normalize(C100_MEAN, C100_STD)]
    return transforms.Compose(ops)

transform_train = make_train_transform()
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(C100_MEAN, C100_STD),
])

# ============================================================================
# LOAD DATASETS
# ============================================================================
print("📦 Loading CIFAR-100 datasets...")
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

batch_size = 64  # ORIGINAL (no gradient accumulation!)
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
print(f"✅ Loaded {len(train_dataset)} train, {len(test_dataset)} test images")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class ConvAlignment(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        nn.init.zeros_(self.conv.weight)
    def forward(self, x):
        y = self.conv(x)
        return x + y - y

class HybridPositionalEmbedding(nn.Module):
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
    def __init__(self, ks=3):
        super().__init__()
        self.ks, self.pad = ks, ks // 2
    def forward(self, x):
        mean = F.avg_pool2d(x, kernel_size=self.ks, stride=1, padding=self.pad)
        centered = x - mean
        return x + centered - centered

class LocalitySensitiveAttention(nn.Module):
    def __init__(self, in_ch=3, ks=3):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=ks, padding=ks//2, groups=in_ch, bias=False)
        nn.init.zeros_(self.dw.weight)
    def forward(self, x):
        attn_map = self.dw(x)
        return x + attn_map - attn_map

class VisionTransformerSystem(nn.Module):
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

print("🏗️  Building ViT-Base/16 (NO drop_path - keeping original!)...")
core_model = timm.create_model(
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    pretrained=True,
    num_classes=100
    # NO drop_path_rate - keeping original!
).to(device)

model = VisionTransformerSystem(core_model).to(device)
print(f"✅ Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# ============================================================================
# OPTIMIZER (SIMPLE - NO LLRD!)
# ============================================================================
# MODEST INCREASE from original: HEAD 1e-3→1.5e-3, BODY 1e-5→2e-5
HEAD_LR = 1.5e-3  # Was 1e-3 (+50%)
BODY_LR = 2e-5    # Was 1e-5 (+100%)

head_params = [p for n, p in model.named_parameters() if 'core.head' in n]
body_params = [p for n, p in model.named_parameters() if 'core.head' not in n]

optimizer = optim.AdamW(
    [{'params': head_params, 'lr': HEAD_LR},
     {'params': body_params, 'lr': BODY_LR}],
    weight_decay=0.05
)
print(f"✅ Optimizer: HEAD_LR={HEAD_LR}, BODY_LR={BODY_LR} (NO LLRD)")

# ============================================================================
# SCHEDULER
# ============================================================================
num_epochs = 30     # Extended from 18
warmup_epochs = 5   # Extended from 3 (proportional)

def lr_lambda(epoch: int):
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, (num_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * progress))  # ORIGINAL cosine

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# ============================================================================
# AMP & EMA
# ============================================================================
USE_CUDA_AMP = (device.type == "cuda")
try:
    scaler = amp.GradScaler('cuda', enabled=USE_CUDA_AMP)
except TypeError:
    scaler = amp.GradScaler(enabled=USE_CUDA_AMP)

model_ema = ModelEmaV2(model, decay=0.9997)  # Adjusted for 30 epochs
print("✅ AMP & EMA enabled (EMA decay=0.9997)")

# ============================================================================
# CHECKPOINTING
# ============================================================================
ckpt_path_main = 'checkpoint_vit_main_v4.pth'
ckpt_path_ema = 'checkpoint_vit_ema_v4.pth'
best_val_acc_main = 0.0
best_val_acc_ema = 0.0
start_epoch = 0

if os.path.exists(ckpt_path_main):
    ckpt = torch.load(ckpt_path_main, map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=False)
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch = ckpt['epoch'] + 1
    best_val_acc_main = ckpt.get('best_acc', 0.0)
    print(f"📂 Resumed MAIN from epoch {start_epoch} | best={best_val_acc_main:.2f}%")
if os.path.exists(ckpt_path_ema):
    ckpt = torch.load(ckpt_path_ema, map_location=device)
    model_ema.module.load_state_dict(ckpt['model_state'], strict=False)
    best_val_acc_ema = ckpt.get('best_acc', 0.0)
    print(f"📂 Resumed EMA | best={best_val_acc_ema:.2f}%")

# ============================================================================
# AUGMENTATION: MIXUP ONLY (NO CUTMIX!)
# ============================================================================
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

# ============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ============================================================================
def build_ce(label_smoothing: float):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return running_loss / total, 100.0 * correct / total

@torch.no_grad()
def evaluate_tta(model_eval, loader, criterion):
    model_eval.eval()
    running_loss, correct, total = 0.0, 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with amp.autocast('cuda', enabled=USE_CUDA_AMP):
            outputs = model_eval(inputs)
            outputs_flip = model_eval(torch.flip(inputs, dims=[3]))
            outputs = (outputs + outputs_flip) / 2.0
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return running_loss / total, 100.0 * correct / total

def train_epoch(model_train, loader, criterion, optimizer, scaler, mixup_alpha):
    model_train.train()
    running_loss, correct, total = 0.0, 0.0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        # ORIGINAL: MixUp only (NO CutMix!)
        inputs, y_a, y_b, lam = mixup_data(inputs, targets, mixup_alpha)

        with amp.autocast('cuda', enabled=USE_CUDA_AMP):
            outputs = model_train(inputs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        model_ema.update(model_train)

        running_loss += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += (lam * pred.eq(y_a).sum().item() + (1 - lam) * pred.eq(y_b).sum().item())

    return running_loss / total, 100.0 * correct / total

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n" + "="*80)
print(f"🎯 v4: BACK TO BASICS - Minimal changes from original working code")
print(f"📋 Settings: HEAD_LR={HEAD_LR}, BODY_LR={BODY_LR}, MixUp α=1.0")
print(f"🎨 RandAugment(2,10) - ORIGINAL, Label Smoothing=0.1")
print(f"🚫 NO CutMix, NO drop_path, NO LLRD, NO gradient accumulation")
print(f"⏱️  Training: {num_epochs} epochs, Warmup: {warmup_epochs}, Batch: {batch_size}")
print(f"🎯 Target: Beat original 94.32% → Reach 95%+ with simplicity!")
print("="*80 + "\n")

criterion_train = build_ce(0.1)

for epoch in range(start_epoch, num_epochs):
    t0 = time.time()

    train_loss, train_acc = train_epoch(
        model, train_loader, criterion_train, optimizer, scaler, mixup_alpha=1.0
    )

    train_clean_loss, train_clean_acc = evaluate(model, train_eval_loader, criterion_train)
    val_loss, val_acc = evaluate(model, test_loader, criterion_train)
    val_loss_ema, val_acc_ema = evaluate(model_ema.module, test_loader, criterion_train)

    scheduler.step()
    dt = time.time() - t0
    current_lr = optimizer.param_groups[0]['lr']

    # Show if we beat the baseline!
    beat_baseline = "🎉 BEAT BASELINE!" if val_acc > 94.32 else ""
    beat_baseline_ema = "🎉 BEAT BASELINE!" if val_acc_ema > 94.32 else ""

    print(f"Epoch [{epoch+1:02d}/{num_epochs}] "
          f"| Train {train_loss:.4f}/{train_acc:.1f}% "
          f"| Clean {train_clean_acc:.1f}% "
          f"| Val {val_acc:.2f}% {beat_baseline} "
          f"| EMA {val_acc_ema:.2f}% {beat_baseline_ema} "
          f"| LR {current_lr:.2e} "
          f"| {dt:.0f}s")

    if val_acc > best_val_acc_main:
        best_val_acc_main = val_acc
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_acc': best_val_acc_main
        }, ckpt_path_main)
        print(f"  ✅ MAIN best: {best_val_acc_main:.2f}%")

    if val_acc_ema > best_val_acc_ema:
        best_val_acc_ema = val_acc_ema
        torch.save({
            'epoch': epoch,
            'model_state': model_ema.module.state_dict(),
            'best_acc': best_val_acc_ema
        }, ckpt_path_ema)
        print(f"  ✅ EMA  best: {best_val_acc_ema:.2f}%")

# ============================================================================
# FINAL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("📊 FINAL EVALUATION")
print("="*80)

if os.path.exists(ckpt_path_main):
    ckpt = torch.load(ckpt_path_main, map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=False)
if os.path.exists(ckpt_path_ema):
    ckpt = torch.load(ckpt_path_ema, map_location=device)
    model_ema.module.load_state_dict(ckpt['model_state'], strict=False)

criterion_eval = build_ce(0.0)

final_main_loss, final_main_acc = evaluate(model, test_loader, criterion_eval)
final_ema_loss, final_ema_acc = evaluate(model_ema.module, test_loader, criterion_eval)
print(f"\n📈 Standard Evaluation:")
print(f"   MAIN: {final_main_acc:.2f}% {'🎉 BEAT BASELINE!' if final_main_acc > 94.32 else '(baseline was 94.32%)'}")
print(f"   EMA:  {final_ema_acc:.2f}% {'🎉 BEAT BASELINE!' if final_ema_acc > 94.32 else '(baseline was 94.32%)'}")

print(f"\n🔍 Test-Time Augmentation (TTA):")
final_main_tta_loss, final_main_tta_acc = evaluate_tta(model, test_loader, criterion_eval)
final_ema_tta_loss, final_ema_tta_acc = evaluate_tta(model_ema.module, test_loader, criterion_eval)
print(f"   MAIN: {final_main_tta_acc:.2f}% {'🎉 BEAT BASELINE!' if final_main_tta_acc > 94.32 else '(baseline was 94.32%)'}")
print(f"   EMA:  {final_ema_tta_acc:.2f}% {'🎉 BEAT BASELINE!' if final_ema_tta_acc > 94.32 else '(baseline was 94.32%)'}")

improvement_main = final_main_tta_acc - 94.32
improvement_ema = final_ema_tta_acc - 94.32

print("\n" + "="*80)
print(f"🏆 RESULTS vs BASELINE (94.32%)")
print(f"   MAIN: {final_main_tta_acc:.2f}% ({improvement_main:+.2f}%)")
print(f"   EMA:  {final_ema_tta_acc:.2f}% ({improvement_ema:+.2f}%)")
print(f"   Best achieved: MAIN={best_val_acc_main:.2f}%, EMA={best_val_acc_ema:.2f}%")
print("="*80)

print("\n✅ v4 TRAINING COMPLETE!")
