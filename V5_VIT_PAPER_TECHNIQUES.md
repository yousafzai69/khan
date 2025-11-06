# v5: Vision Transformer - Following "An Image is Worth 16x16 Words"

## 📄 Paper Reference

**Title**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
**Authors**: Alexey Dosovitskiy et al. (Google Research)
**Published**: ICLR 2021
**arXiv**: 2010.11929

This is the foundational paper that introduced Vision Transformers (ViT) to computer vision, demonstrating that pure transformers can match or exceed CNN performance on image classification.

## 🎯 Core Insight from the Paper

> "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. **We show that reliance on CNNs is not necessary** and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks."

## 🔑 Key Techniques from the Paper (Implemented in v5)

### 1. **Pure Transformer Architecture**

**Paper's Innovation**: No CNNs! Images → Patches → Transformer

```
Traditional CNN approach:
Image → Conv layers → Features → Classifier

ViT approach (Paper):
Image → Patches → Linear embedding → Transformer → Classifier
```

**v5 Implementation**:
```python
# Using pure ViT-Base/16 from timm (no CNN wrappers)
model = timm.create_model(
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    pretrained=True,
    num_classes=100
)
```

**Why it works**: Transformers learn long-range dependencies naturally through self-attention, which CNNs struggle with due to limited receptive fields.

---

### 2. **Patch Embedding: "An Image is Worth 16x16 Words"**

**Paper's Approach**:
- Divide image into fixed-size patches (16×16 for ImageNet)
- Flatten each patch into a vector
- Linearly project to embedding dimension
- Treat patches like "words" in NLP

**Math**:
```
Image: H×W×C
Patches: N = (H/P)×(W/P) where P=patch size
Each patch: P×P×C → Flatten → D-dimensional embedding
```

**For CIFAR-100**:
- Original images: 32×32 (too small!)
- Resize to 224×224 (match pre-training)
- Patch 16×16 → 196 patches (14×14 grid)
- Each patch → 768-dimensional embedding

**Research Finding**: For small datasets like CIFAR, **4×4 patches work better** than 16×16, but we use 16×16 to match pre-trained weights.

---

### 3. **Pre-training + Fine-tuning (Two-Stage Approach)**

**Paper's Key Finding**:
> "Large scale training trumps inductive bias. Vision Transformer attains excellent results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints."

**Training Strategy**:
1. **Pre-training**: Train on large dataset (ImageNet-21k or JFT-300M)
2. **Fine-tuning**: Adapt to target task (CIFAR-100)

**v5 Implementation**:
```python
# Use ImageNet-21k pre-trained weights (11M images, 21k classes)
model = timm.create_model(
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    pretrained=True,  # Load pre-trained weights
    num_classes=100    # Replace head for CIFAR-100
)
```

**Why it works**: Pre-training learns general visual representations, fine-tuning specializes for CIFAR-100.

---

### 4. **Position Embeddings**

**Problem**: Transformers have no notion of spatial position!

**Paper's Solution**: Add learned position embeddings

```python
# Conceptually (built into ViT):
patch_embeddings = Linear(patches)  # [N, 768]
position_embeddings = LearnedParam([N, 768])
embeddings = patch_embeddings + position_embeddings
```

**Types tried in paper**:
1. 1D position embeddings (what we use)
2. 2D position embeddings
3. Relative position embeddings

**Finding**: "1D position embeddings work well" - spatial structure emerges naturally!

---

### 5. **[CLS] Token for Classification**

**From BERT (NLP)**: Prepend special [CLS] token, use its output for classification

**ViT adapts this**:
```python
# Conceptually:
cls_token = LearnedParam([1, 768])
sequence = [cls_token, patch1, patch2, ..., patchN]
transformer_output = Transformer(sequence)
classification = MLP(transformer_output[0])  # Use [CLS] token
```

**v5**: Built into timm's ViT implementation

**Why it works**: [CLS] token aggregates information from all patches via self-attention.

---

### 6. **Adam Optimizer with Specific Hyperparameters**

**Paper's Training Setup (for ImageNet pre-training)**:
- Optimizer: Adam
- β1 = 0.9
- β2 = 0.999
- Batch size: 4096 (large!)
- Weight decay: 0.1
- Learning rate: Varies with warmup

**v5 for CIFAR-100 Fine-tuning**:
```python
optimizer = optim.Adam(
    [{'params': head_params, 'lr': 1e-3},   # Higher for new head
     {'params': body_params, 'lr': 5e-5}],  # Lower for pre-trained body
    betas=(0.9, 0.999),  # Paper's β values
    weight_decay=0.1      # Paper's weight decay
)
```

---

### 7. **Learning Rate Schedule with Warmup**

**Paper's Schedule**:
1. Linear warmup (gradually increase LR)
2. Cosine decay (smoothly decrease to minimum)

**v5 Implementation**:
```python
class WarmupCosineScheduler:
    def step(self, epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup) / (total - warmup)
            lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

**Why it works**:
- Warmup: Prevents early training instability
- Cosine decay: Smooth convergence to optimum

---

### 8. **Model Architecture: ViT-Base**

**Paper's ViT-Base Configuration**:
```
Layers (L): 12
Hidden dimension (D): 768
MLP dimension: 3072 (4×D)
Attention heads (H): 12
Patch size (P): 16
Parameters: 86M
```

**Comparison to BERT**:
- ViT-Base ≈ BERT-Base (architecture)
- Paper also defines ViT-Large and ViT-Huge

**v5 uses**: ViT-Base/16 (optimal for CIFAR-100 after fine-tuning)

---

## 🔬 CIFAR-100 Specific Findings (from Research)

### Paper's Results on CIFAR-100:
- **94.55% accuracy** (with ImageNet-21k pre-training)
- Better than ResNets of similar size
- Benefits from pre-training even on small datasets

### Additional Research Findings for CIFAR-100:

| Finding | Value | Source |
|---------|-------|--------|
| **Optimal batch size** | 64 (not 4096!) | Community research |
| **Optimal patch size** | 4×4 > 16×16 | For 32×32 images |
| **Model size** | Smaller is better | Small datasets prefer smaller models |
| **Augmentation** | Less for short schedules | <7 epochs: skip augmentation |
| **Training duration** | 7-30 epochs | Short fine-tuning sufficient |

### v5 Configuration (Optimized for CIFAR-100):

```python
# Model
Model: ViT-Base/16 (86M params)
Pre-trained: ImageNet-21k
Patch size: 16×16 (match pre-training)
Image size: 224×224 (resize from 32×32)

# Training
Epochs: 30
Batch size: 64 (optimal for CIFAR)
Warmup: 5 epochs
Optimizer: Adam (β1=0.9, β2=0.999)
Weight decay: 0.1
LR (head): 1e-3
LR (body): 5e-5

# Augmentation (moderate for fine-tuning)
RandAugment: (2, 9)
MixUp: α=0.8
Label smoothing: 0.1
```

---

## 📊 Expected Results

### Paper's CIFAR-100 Performance:
- **ViT-Base/16 (ImageNet-21k pre-trained): 94.55%**
- ViT-Large: Higher but more expensive

### v5 Target:
- **MAIN**: 94.5-95.0%
- **EMA**: 95.0-95.5%
- **EMA + TTA**: 95.5-96.0%

**Strategy**: Match or beat paper's results using optimal fine-tuning!

---

## 🆚 Comparison: v5 vs Previous Versions

| Feature | v2 | v3 | v4 | **v5 (ViT Paper)** |
|---------|----|----|----|--------------------|
| **Philosophy** | "Improve" everything | Fix v2 | Back to basics | **Follow paper** |
| **Architecture** | ViT + custom layers | ViT + custom | ViT + custom | **Pure ViT** |
| **Optimizer** | AdamW + LLRD | AdamW + LLRD | AdamW simple | **Adam (paper)** |
| **β values** | Default | Default | Default | **0.9, 0.999** |
| **Weight decay** | 0.05 | 0.05 | 0.05 | **0.1 (paper)** |
| **Batch size** | 256 (accum) | 256 (accum) | 64 | **64 (optimal)** |
| **Augmentation** | Strong | Moderate | Weak | **Moderate** |
| **Schedule** | Cosine + min | Cosine + min | Cosine | **Warmup + Cosine** |
| **Source** | Mixed ideas | Fix over-reg | Original code | **ViT paper** |

**v5 Advantages**:
1. ✅ Follows proven paper methodology
2. ✅ Pure ViT (no custom CNN layers)
3. ✅ Optimal hyperparameters from paper
4. ✅ CIFAR-100 specific optimizations
5. ✅ Clean, principled approach

---

## 🎓 Key Insights from the Paper

### 1. **Inductive Bias Trade-off**

**CNNs have strong inductive biases**:
- Translation equivariance
- Locality (nearby pixels matter)
- Scale invariance

**ViT has weak inductive bias**:
- Only position embeddings
- Must learn spatial relationships
- **Needs more data** to compensate

**Trade-off**:
```
Small data: CNNs > ViT (inductive bias helps)
Large data: ViT ≥ CNNs (learns patterns from data)
```

**Solution for CIFAR-100**: Pre-train on large data (ImageNet-21k), then fine-tune!

### 2. **Scaling Law**

Paper shows:
```
Performance ∝ log(Pre-training dataset size)
```

More pre-training data → Better fine-tuning performance!

### 3. **Position Embeddings Learn Spatial Structure**

Paper's visualization shows:
- Position embeddings encode 2D grid structure
- Similar positions have similar embeddings
- Spatial relationships emerge naturally

**Implication**: ViT learns "where patches are" without explicit 2D encoding!

### 4. **Attention Distance vs CNN Receptive Field**

Paper analyzes attention patterns:
- Early layers: Local attention (like CNN low layers)
- Later layers: Global attention (unlike CNNs!)
- [CLS] token: Attends to entire image

**Advantage over CNNs**: ViT captures long-range dependencies from layer 1!

---

## 🚀 Why v5 Should Perform Well

1. **Proven Architecture**: ViT paper achieved 94.55% on CIFAR-100
2. **Optimal Hyperparameters**: Following paper's recommendations
3. **Pre-training**: Leveraging ImageNet-21k learned representations
4. **CIFAR-100 Optimizations**: Batch size 64, moderate augmentation
5. **Simplicity**: Pure ViT, no unnecessary complexity

---

## 📚 Additional Paper Insights

### Computational Cost

**FLOPs for ViT-Base**:
```
Per image: ~17.6 GFLOPs
vs ResNet-50: ~4.1 GFLOPs
```

**Trade-off**: ViT is 4×slower but more accurate with pre-training!

### Robustness

Paper shows ViT is more robust to:
- Image perturbations
- Adversarial attacks
- Distribution shifts

**Reason**: Global self-attention vs local CNN filters

### Interpretability

ViT attention maps are more interpretable:
- Can visualize which patches attend to each other
- [CLS] token attention shows "what model looks at"
- Patch embeddings cluster by visual similarity

---

## 🎯 Summary

**v5 implements the pure Vision Transformer approach from the seminal paper**:

✅ Pure transformer (no CNN bias)
✅ Patch embeddings (16×16)
✅ Pre-training + fine-tuning
✅ Adam optimizer (β1=0.9, β2=0.999)
✅ Weight decay 0.1
✅ Batch size 64 (optimal for CIFAR)
✅ Warmup + cosine schedule
✅ Moderate augmentation

**Target**: Beat 94.32% baseline → Reach 95%+ using proven ViT principles!

**Philosophy**: "An Image is Worth 16x16 Words" - treat vision like NLP!

---

## 📖 References

1. Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ICLR 2021
2. Paper's CIFAR-100 result: 94.55%
3. Community research on CIFAR-100 ViT fine-tuning best practices
4. timm library implementation: `vit_base_patch16_224.augreg_in21k_ft_in1k`
