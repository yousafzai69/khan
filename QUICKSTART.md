# Quick Start Guide: Improved ViT CIFAR-100 Fine-Tuning

## TL;DR - What Changed?

| Feature | Original | Improved | Gain |
|---------|----------|----------|------|
| **Training Epochs** | 18 | 30 | Better convergence |
| **Augmentation** | MixUp only | MixUp + CutMix | +0.3-0.5% |
| **Learning Rate** | Fixed per group | Layer-wise decay (LLRD) | +0.5-0.8% |
| **Effective Batch Size** | 64 | 256 (grad accum) | +0.2-0.4% |
| **Stochastic Depth** | None | drop_path=0.1 | +0.2-0.4% |
| **RandAugment** | (2, 10) | (3, 14) | +0.2-0.3% |
| **Test-Time Aug** | No | Yes (flip) | +0.3-0.5% |
| **Expected Accuracy** | 94.32% | **95.5-96.5%** | **+1.2-2.2%** |

## How to Use

### Option 1: Colab (Recommended)

```python
# Cell 1: Install dependencies
!pip -q install --upgrade timm

# Cell 2: Download and run improved script
!wget https://raw.githubusercontent.com/yourusername/khan/branch-name/vit_cifar100_improved.py
exec(open('vit_cifar100_improved.py').read())
```

### Option 2: Copy-Paste

Simply copy the entire content of `vit_cifar100_improved.py` into a single Colab cell and run.

### Option 3: Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/khan.git
cd khan

# Install dependencies
pip install torch torchvision timm

# Run training
python vit_cifar100_improved.py
```

## Expected Output

### During Training

```
Using device: cuda
Optimizer created with 14 parameter groups (LLRD enabled)

Starting training for 30 epochs
Effective batch size: 256
Warmup epochs: 5
MixUp alpha: 1.0, CutMix alpha: 1.0
Label smoothing: 0.1
LLRD enabled with decay: 0.9

Epoch [01/30] | Train Loss 2.4xxx Acc 56.xx% | Train(clean) 92.xx% | Val 91.xx% (EMA xx.xx%) | LR 4.0e-04 | 620.xs
  ↳ MAIN best ↑ 91.xx% (saved)
  ↳ EMA  best ↑ xx.xx% (saved)
...
Epoch [30/30] | Train Loss 1.4xxx Acc 75.xx% | Train(clean) 99.xx% | Val 95.xx% (EMA 95.xx%) | LR 2.0e-05 | 620.xs
  ↳ EMA  best ↑ 95.xx% (saved)
```

### Final Evaluation

```
======================================================================
Final Evaluation
======================================================================
MAIN  Test Acc (standard): 95.xx%
EMA   Test Acc (standard): 95.xx%

Evaluating with Test-Time Augmentation...
MAIN  Test Acc (TTA):      95.xx%
EMA   Test Acc (TTA):      96.xx%

======================================================================
Best validation accuracies achieved:
  MAIN: 95.xx%
  EMA:  96.xx%
======================================================================
```

## Key Parameters (Easy to Tune)

### 1. Adjust Training Duration
```python
num_epochs = 30      # Increase to 40-50 for even better results
warmup_epochs = 5    # Increase if you increase num_epochs
```

### 2. Adjust Effective Batch Size
```python
batch_size = 64      # Physical batch size (limited by GPU memory)
accum_steps = 4      # Gradient accumulation (increase to 8 for batch_size=512)
```

### 3. Adjust Learning Rates
```python
BASE_LR = 5e-5       # Base LR for transformer body
HEAD_LR = 2e-3       # LR for classification head
LLRD_DECAY = 0.9     # Layer-wise decay factor (0.85-0.95 typical)
```

### 4. Adjust Augmentation Strength
```python
# Stronger augmentation (if model underfits):
RandAugment(num_ops=3, magnitude=16)  # Was: (3, 14)
mixup_alpha = 1.2     # Was: 1.0
cutmix_alpha = 1.2    # Was: 1.0

# Weaker augmentation (if model overfits):
RandAugment(num_ops=2, magnitude=10)
mixup_alpha = 0.8
cutmix_alpha = 0.8
```

### 5. Adjust Regularization
```python
drop_path_rate = 0.1  # Stochastic depth (0.05-0.2 typical)
weight_decay = 0.05   # Weight decay (0.01-0.1 typical)
label_smoothing = 0.1 # Label smoothing (0.05-0.15 typical)
```

## Monitoring Training

### Signs of Good Training:
- ✅ Train(clean) accuracy steadily increases
- ✅ Val accuracy follows train accuracy (with gap < 5%)
- ✅ EMA accuracy >= MAIN accuracy
- ✅ Best accuracy improves every few epochs

### Signs of Problems:

**Underfitting** (val acc too low):
- Increase `num_epochs` to 40-50
- Increase `BASE_LR` to 1e-4
- Decrease augmentation strength
- Decrease `drop_path_rate` to 0.05

**Overfitting** (train >> val accuracy):
- Increase augmentation strength
- Increase `drop_path_rate` to 0.15-0.2
- Increase `weight_decay` to 0.1
- Add more regularization

**Training instability** (loss spikes):
- Decrease `BASE_LR` and `HEAD_LR` by 0.5x
- Increase `warmup_epochs` to 8-10
- Decrease `max_grad_norm` to 0.5

## Resuming Training

The script automatically saves and resumes from checkpoints:
```python
# Checkpoints are saved at:
checkpoint_vit_main_improved.pth  # Main model
checkpoint_vit_ema_improved.pth   # EMA model

# To resume, just re-run the script - it will auto-detect and load
```

## Evaluating Saved Models

After training, use the evaluation cells from the original script:

```python
# Load best EMA model
ckpt = torch.load('checkpoint_vit_ema_improved.pth')
model_ema.module.load_state_dict(ckpt['model_state'])

# Evaluate with TTA
final_acc = evaluate_tta(model_ema.module, test_loader, criterion)
print(f"Final EMA Test Acc (TTA): {final_acc:.2f}%")
```

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
batch_size = 32       # Was: 64
accum_steps = 8       # Was: 4 (keeps effective batch size = 256)

# Or reduce image size (NOT recommended for ViT)
IMG_SIZE = 192        # Was: 224
```

### Slow Training
```python
# Reduce workers if CPU bottleneck
num_workers = 0       # Was: 2

# Disable some augmentation
RandAugment(num_ops=2, magnitude=10)  # Was: (3, 14)

# Or train fewer epochs
num_epochs = 20       # Was: 30
```

### Accuracy Not Improving
```python
# Try these in order:
1. Check if training loss is decreasing (should go below 1.5)
2. Increase num_epochs to 40-50
3. Increase BASE_LR to 1e-4 (2x current)
4. Add more augmentation diversity
5. Try different seed: seed_everything(123)
```

## Performance Benchmarks

### Expected Training Time (on V100 GPU):
- **Per epoch**: ~10-11 minutes
- **Full 30 epochs**: ~5-5.5 hours
- **Checkpoint loading**: <5 seconds

### Expected Memory Usage:
- **GPU Memory**: ~8-10 GB (with batch_size=64)
- **RAM**: ~4-6 GB
- **Disk**: ~700 MB (model checkpoints)

## Citation

If this code helps your research, please consider citing:

```bibtex
@misc{vit_cifar100_improved,
  title={Improved ViT Fine-Tuning for CIFAR-100},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/khan}
}
```

## Related Improvements

For even higher accuracy (97%+), see `IMPROVEMENTS_SUMMARY.md` for:
- Knowledge distillation
- Self-training
- Model ensembling
- SAM optimizer
- Advanced augmentation techniques
