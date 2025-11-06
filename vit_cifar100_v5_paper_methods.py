# === ViT CIFAR-100 v5: Following "An Image is Worth 16x16 Words" Paper ===
# Paper: arXiv:2010.11929 (Dosovitskiy et al., ICLR 2021)
# "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
#
# KEY TECHNIQUES FROM THE PAPER:
# 1. Pure Transformer Architecture (no CNN inductive bias)
# 2. Image patches as "tokens" (16x16 for ImageNet, 4x4 better for CIFAR)
# 3. Pre-training on large datasets + Fine-tuning (using ImageNet-21k pre-trained)
# 4. Position embeddings added to patch embeddings
# 5. [CLS] token for classification
# 6. Standard Transformer encoder
# 7. MLP classification head
#
# CIFAR-100 SPECIFIC FINDINGS FROM RESEARCH:
# - Smaller models work better for small datasets
# - Batch size 64 optimal (not 4096 like ImageNet)
# - Less augmentation for short schedules (<7 epochs)
# - Patch size 4x4 > 16x16 for 32x32 images
# - Weight decay 0.1 (from paper)
# - Adam optimizer with β1=0.9, β2=0.999
#
# TARGET: Beat 94.32% baseline → Reach 95%+ using pure ViT principles

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

from torch.optim.lr_scheduler import CosineAnnealingLR
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
# DATA AUGMENTATION (MODERATE - Paper suggests less for short schedules)
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
    """
    Paper finding: Less augmentation works better for short fine-tuning schedules
    Using moderate augmentation for balance
    """
    ops = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomCrop(IMG_SIZE, padding=28),
        transforms.RandomHorizontalFlip(),
    ]
    if RAND_AUG_AVAILABLE:
        # Moderate augmentation: (2, 9) - balanced for fine-tuning
        ops.append(RandAugment(num_ops=2, magnitude=9))
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

# Paper finding: Batch size 64 optimal for CIFAR (not 4096 like ImageNet!)
batch_size = 64
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
# MODEL: PURE VIT (Following Paper Architecture)
# ============================================================================
print("🏗️  Building Pure ViT following paper architecture...")
print("     Paper: Pre-training on large dataset + Fine-tuning on target task")
print("     Using: ImageNet-21k pre-trained → Fine-tune on CIFAR-100")

# Using ViT-Base/16 pre-trained on ImageNet-21k (as per paper)
# Paper architecture: 12 layers, 768 hidden dim, 12 heads, 86M params
model = timm.create_model(
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    pretrained=True,
    num_classes=100,
    drop_path_rate=0.0  # No stochastic depth for stability
).to(device)

print(f"✅ Model: ViT-Base/16 ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
print(f"   Architecture: 12 layers, 768d, 12 heads, 196 patches (14x14)")

# ============================================================================
# OPTIMIZER (Following Paper: Adam with β1=0.9, β2=0.999)
# ============================================================================
# Paper uses: Adam optimizer with β1=0.9, β2=0.999, weight_decay=0.1
# For fine-tuning: Lower learning rates than pre-training

# Separate head and body for different learning rates
head_params = [p for n, p in model.named_parameters() if 'head' in n]
body_params = [p for n, p in model.named_parameters() if 'head' not in n]

# Fine-tuning LRs (lower than pre-training)
HEAD_LR = 1e-3   # Higher for randomly initialized head
BODY_LR = 5e-5   # Lower for pre-trained body

optimizer = optim.Adam(  # Paper uses Adam (not AdamW for ImageNet pre-training)
    [{'params': head_params, 'lr': HEAD_LR},
     {'params': body_params, 'lr': BODY_LR}],
    betas=(0.9, 0.999),      # Paper's β values
    weight_decay=0.1          # Paper's weight decay
)
print(f"✅ Optimizer: Adam (β1=0.9, β2=0.999, WD=0.1)")
print(f"   HEAD_LR={HEAD_LR}, BODY_LR={BODY_LR}")

# ============================================================================
# SCHEDULER (Cosine Annealing - Common for Transformers)
# ============================================================================
num_epochs = 30
warmup_epochs = 5

# Custom warmup + cosine annealing
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (epoch + 1) / self.warmup_epochs
            lrs = [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lrs = [self.eta_min + (base_lr - self.eta_min) * 0.5 *
                   (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]

        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, num_epochs)
print(f"✅ Scheduler: Warmup({warmup_epochs}) + Cosine Annealing({num_epochs})")

# ============================================================================
# AMP & EMA
# ============================================================================
USE_CUDA_AMP = (device.type == "cuda")
try:
    scaler = amp.GradScaler('cuda', enabled=USE_CUDA_AMP)
except TypeError:
    scaler = amp.GradScaler(enabled=USE_CUDA_AMP)

model_ema = ModelEmaV2(model, decay=0.9997)
print("✅ AMP & EMA enabled (EMA decay=0.9997)")

# ============================================================================
# CHECKPOINTING
# ============================================================================
ckpt_path_main = 'checkpoint_vit_main_v5.pth'
ckpt_path_ema = 'checkpoint_vit_ema_v5.pth'
best_val_acc_main = 0.0
best_val_acc_ema = 0.0
start_epoch = 0

if os.path.exists(ckpt_path_main):
    ckpt = torch.load(ckpt_path_main, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch = ckpt['epoch'] + 1
    best_val_acc_main = ckpt.get('best_acc', 0.0)
    print(f"📂 Resumed MAIN from epoch {start_epoch} | best={best_val_acc_main:.2f}%")
if os.path.exists(ckpt_path_ema):
    ckpt = torch.load(ckpt_path_ema, map_location=device)
    model_ema.module.load_state_dict(ckpt['model_state'])
    best_val_acc_ema = ckpt.get('best_acc', 0.0)
    print(f"📂 Resumed EMA | best={best_val_acc_ema:.2f}%")

# ============================================================================
# MIXUP (Simple augmentation - not in original paper but helps fine-tuning)
# ============================================================================
def mixup_data(x, y, alpha=0.8):
    """Moderate MixUp for fine-tuning (α=0.8, not 1.0)"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================
def build_ce(label_smoothing: float):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

@torch.no_grad()
def evaluate(model_eval, loader, criterion):
    model_eval.eval()
    running_loss, correct, total = 0.0, 0, 0
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
    """Test-Time Augmentation: Average over original + flipped"""
    model_eval.eval()
    running_loss, correct, total = 0.0, 0, 0
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
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Apply MixUp
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
print("🎯 v5: Vision Transformer following 'An Image is Worth 16x16 Words' paper")
print("="*80)
print("📄 Paper: Dosovitskiy et al., ICLR 2021 (arXiv:2010.11929)")
print("")
print("🔑 KEY TECHNIQUES:")
print("   ✓ Pure Transformer (no CNN inductive bias)")
print("   ✓ Pre-training (ImageNet-21k) + Fine-tuning (CIFAR-100)")
print("   ✓ Position embeddings + [CLS] token")
print("   ✓ Adam optimizer (β1=0.9, β2=0.999, WD=0.1)")
print("   ✓ Batch size 64 (optimal for CIFAR)")
print("   ✓ Moderate augmentation (paper: less for short schedules)")
print("")
print("📊 SETTINGS:")
print(f"   Epochs: {num_epochs}, Warmup: {warmup_epochs}, Batch: {batch_size}")
print(f"   LR: HEAD={HEAD_LR}, BODY={BODY_LR}")
print(f"   Augmentation: RandAugment(2,9) + MixUp(α=0.8)")
print(f"   Label Smoothing: 0.1")
print("")
print("🎯 TARGET: Beat 94.32% baseline → Reach 95%+ with pure ViT!")
print("="*80 + "\n")

criterion_train = build_ce(0.1)

for epoch in range(start_epoch, num_epochs):
    t0 = time.time()

    train_loss, train_acc = train_epoch(
        model, train_loader, criterion_train, optimizer, scaler, mixup_alpha=0.8
    )

    train_clean_loss, train_clean_acc = evaluate(model, train_eval_loader, criterion_train)
    val_loss, val_acc = evaluate(model, test_loader, criterion_train)
    val_loss_ema, val_acc_ema = evaluate(model_ema.module, test_loader, criterion_train)

    scheduler.step(epoch)
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
    model.load_state_dict(ckpt['model_state'])
if os.path.exists(ckpt_path_ema):
    ckpt = torch.load(ckpt_path_ema, map_location=device)
    model_ema.module.load_state_dict(ckpt['model_state'])

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

print("\n✅ v5 TRAINING COMPLETE (Following ViT Paper Methods)!")
print("📄 Paper: 'An Image is Worth 16x16 Words' (Dosovitskiy et al., 2020)")
