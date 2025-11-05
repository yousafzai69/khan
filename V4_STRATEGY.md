# v4 Strategy: Back to Basics - Why Less is More

## 🚨 The Problem with v3

**v3 MAIN model accuracy: 93.56%** (0.76% BELOW the original 94.32% baseline!)

Despite "improvements", v3 actually made MAIN worse than the original working code.

## 🔍 Root Cause: Over-Regularization (AGAIN!)

v3 claimed to "fix over-regularization" but was STILL over-regularized compared to the original:

| Setting | **Original (94.32%)** ✅ | **v3 (93.56%)** ❌ | Change |
|---------|-------------------------|-------------------|---------|
| **RandAugment** | (2, 10) | (2, 12) | +20% stronger |
| **MixUp/CutMix** | MixUp only | 70% MixUp + 30% CutMix | Added CutMix |
| **Drop Path** | **0.0** | 0.05 | Added dropout |
| **HEAD_LR** | 1e-3 | 2.5e-3 | 2.5x higher |
| **BODY_LR** | 1e-5 | 8e-5 | 8x higher |
| **LLRD** | NO | YES (14 groups) | Added complexity |
| **Batch Size** | 64 | 256 (grad accum) | 4x larger |
| **Epochs** | 18 | 30 | 67% more |

### Evidence from v3 Training:
- Epoch 25: Train clean **99.8%**, Val **93.52%** → **6.3% gap = overfitting!**
- MAIN never caught up to baseline despite 25/30 epochs
- EMA barely beat baseline (94.34% vs 94.32%)

## 💡 v4 Solution: Return to Simplicity

**Philosophy**: The original code worked (94.32% in just 18 epochs). Don't "fix" what isn't broken!

### v4 Changes from Original (MINIMAL!):

| Setting | Original | **v4** | Rationale |
|---------|----------|--------|-----------|
| **Epochs** | 18 | **30** | More training time |
| **HEAD_LR** | 1e-3 | **1.5e-3** | +50% modest increase |
| **BODY_LR** | 1e-5 | **2e-5** | +100% modest increase |
| **EMA decay** | 0.9998 | **0.9997** | Adjusted for 30 epochs |
| **Warmup** | 3 | **5** | Proportional to epochs |
| **Everything else** | ✅ | **✅ SAME!** | Keep what works! |

### What v4 DOES NOT Change:
- ❌ NO CutMix (just MixUp α=1.0)
- ❌ NO drop_path (keep 0.0)
- ❌ NO LLRD (simple 2-group optimizer)
- ❌ NO gradient accumulation (batch=64)
- ❌ NO stronger RandAugment (keep 2,10)
- ❌ NO complicated tricks

## 📊 Expected Results

### v3 Trajectory (epoch 25):
- MAIN: 93.56% (❌ below baseline)
- EMA: 94.34% (barely above baseline)
- **Unlikely** to beat 95% by epoch 30

### v4 Expected Results (30 epochs):
Based on original (94.32% in 18 epochs) + 12 more epochs + modest LR increase:

- **MAIN**: 94.5-95.0% (✅ beats baseline)
- **EMA**: 95.0-95.5% (✅ significantly better)
- **With TTA**: +0.3-0.5% boost
- **Final target**: 95.0-95.5%+ EMA with TTA

### Why v4 Should Work:

1. **Original achieved 94.32% in 18 epochs** → Proven formula
2. **30 epochs** (67% more) → Better convergence
3. **Modest LR increase** (+50%/+100%) → Faster learning, not over-aggressive
4. **No over-regularization** → Model can actually learn!
5. **Simplicity** → Less to go wrong

## 🎯 Key Learnings

### 1. "Improvements" Can Make Things Worse
- v2 added 9 "improvements" → 93.85% MAIN, 82.70% EMA (TERRIBLE!)
- v3 "fixed" v2 but still added too much → 93.56% MAIN (below baseline!)
- **Lesson**: Don't add complexity without validation

### 2. Regularization is About Balance
Too little regularization → Overfitting (train >> val)
**Too much regularization → Underfitting (can't reach baseline!)**  ← v2 & v3 here
Sweet spot → Just right

### 3. Original Code Was Already Good!
The original achieved **94.32% in just 18 epochs** with:
- Simple settings
- No fancy tricks
- Proven hyperparameters

**Why mess with success?**

### 4. Occam's Razor Applies to ML
- Simpler models train faster
- Fewer hyperparameters = less to tune
- Easier to debug
- More reproducible

## 🔬 Hypothesis: Why Original Stopped at 18 Epochs

The original code achieved 94.32% at epoch 18 but might have been **STILL IMPROVING!**

Evidence:
- Training loss was still decreasing
- No plateau observed
- Cosine schedule hadn't finished

**v4 test**: Run original settings for 30 epochs → Should reach 95%+ naturally!

## 📋 Comparison Table: All Versions

| Metric | Original | v2 | v3 | **v4** |
|--------|----------|----|----|--------|
| **MAIN Acc** | 94.32% | 93.85% | 93.56% | **Target: 94.5-95%** |
| **EMA Acc** | 94.32% | 82.70% | 94.34% | **Target: 95-95.5%** |
| **vs Baseline** | Baseline | -0.47% / -11.62% | -0.76% / +0.02% | **Expected: +0.2-1.2%** |
| **Epochs** | 18 | 20 | 30 | **30** |
| **RandAug** | (2,10) | (3,14) | (2,12) | **(2,10) ✅** |
| **MixUp/CutMix** | MixUp | 50/50 | 70/30 | **MixUp only ✅** |
| **Drop Path** | 0.0 | 0.1 | 0.05 | **0.0 ✅** |
| **LLRD** | NO | YES | YES | **NO ✅** |
| **Batch Size** | 64 | 256 | 256 | **64 ✅** |
| **HEAD_LR** | 1e-3 | 2e-3 | 2.5e-3 | **1.5e-3** |
| **BODY_LR** | 1e-5 | 5e-5 | 8e-5 | **2e-5** |
| **Complexity** | Low | High | Medium | **Low ✅** |
| **Philosophy** | Simple | "Improve" everything | "Fix" v2 | **Keep what works** |

## 🎬 Action Plan

### Step 1: Run v4 for 30 Epochs
```python
# Use vit_cifar100_v4_minimal.py
# Expected time: ~5 hours on V100
# Expected result: MAIN 94.5-95%, EMA 95-95.5%
```

### Step 2: Monitor Key Metrics
- **MAIN accuracy** should exceed 94.32% by epoch 20-25
- **EMA accuracy** should be 0.3-0.5% higher than MAIN
- **Train clean** vs **Val** gap should be 4-5% (healthy)

### Step 3: If v4 Works...
Then we know the original formula was correct all along!
- Original just needed more epochs
- No fancy tricks required
- Simplicity wins

### Step 4: If v4 Still Underperforms...
Then something else is wrong (unlikely given original worked):
- Check for bugs
- Verify data augmentation
- Test with even weaker augmentation
- Consider hardware/environment differences

## 🏆 Prediction

**v4 will beat v3 because**:
1. Original formula was proven (94.32%)
2. 30 epochs > 18 epochs = better convergence
3. Modest LR increase = faster learning
4. No over-regularization = model can learn
5. Simplicity = fewer things to go wrong

**Final prediction for v4 (30 epochs)**:
- MAIN: **94.7%** (±0.2%)
- EMA: **95.2%** (±0.3%)
- EMA + TTA: **95.5-95.8%**

This would beat the baseline by **+1.2-1.5%** - a solid improvement!

## 📚 Philosophy: Less is More

> "Perfection is achieved, not when there is nothing more to add,
> but when there is nothing left to take away." - Antoine de Saint-Exupéry

Applied to ML:
- Don't add techniques just because they exist
- Validate each change against baseline
- Respect what already works
- Simplicity aids reproducibility

**v4 embodies this philosophy**: Start simple, add only what's proven necessary.

---

**Next**: Run v4 and validate this hypothesis! 🚀
