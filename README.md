# MobilePlantViT-LDA

A Lightweight Hybrid CNN-Transformer Architecture for Plant Disease Classification with Linear Differential Attention

---

## 🌿 Overview

**MobilePlantViT-LDA** is a novel lightweight deep learning architecture designed for efficient plant disease classification on mobile and edge devices. It combines the local feature extraction capabilities of MobileNet-inspired CNNs with the global context understanding of Vision Transformers (ViT), featuring a custom **Linear Differential Attention (LDA)** mechanism for improved noise cancellation and feature extraction.

### Key Features

- **Hybrid Architecture**: Combines efficient CNN backbone with transformer attention
- **Linear Differential Attention**: Novel attention mechanism that computes the difference between two attention maps for noise cancellation
- **Mobile-First Design**: Optimized for deployment on resource-constrained devices (<5M parameters)
- **Multiple Variants**: Tiny (~220K), Small (~490K), Base (~867K), and Large (~1.9M) parameter configurations
- **Comprehensive Pipeline**: End-to-end training, evaluation, and deployment workflow

---

## 🏗️ Architecture

```markdown
Input (224×224×3)
       │
       ▼ CNN Stage
┌──────────────────┐
│    GhostConv     │  → Efficient feature generation with ghost features
├──────────────────┤
│   Fused-IR Block │  → Fused inverted residual for spatial processing
├──────────────────┤
│ Coordinate Attn  │  → Position-aware channel attention
└──────────────────┘
       │
       ▼ Transition Stage
┌──────────────────┐
│  Patch Embedding │  → Convert spatial features to sequence
├──────────────────┤
│ Positional Enc.  │  → Add position information
└──────────────────┘
       │
       ▼ Transformer Stage
┌──────────────────┐
│       LDA        │  → Linear Differential Attention
├──────────────────┤
│   Residual LN    │  → LayerNorm with skip connection
├──────────────────┤
│  Bottleneck FFN  │  → Parameter-efficient feed-forward
└──────────────────┘
       │
       ▼ Classifier Stage
┌──────────────────┐
│       GAP        │  → Global Average Pooling
├──────────────────┤
│  Classifier Head │  → Final classification
└──────────────────┘
       │
       ▼
Output (38 classes)
```



### Linear Differential Attention (LDA)

The core innovation of this architecture is the **Linear Differential Attention** mechanism:

```markdown
A_diff = α × (softmax(Q₁K₁ᵀ) - softmax(Q₂K₂ᵀ))
```



This differential approach:
- Cancels out noise common to both attention maps
- Enhances meaningful patterns that differ between maps
- Provides learnable noise cancellation via the α parameter

---

## 📊 Model Variants

| Variant | Parameters | embed_dim | num_heads | Use Case |
|---------|------------|-----------|-----------|----------|
| **Tiny** | ~220K | 128 | 4 | Edge devices, IoT, real-time inference |
| **Small** | ~490K | 192 | 6 | Mobile apps, balanced performance |
| **Base** | ~867K | 256 | 8 | Default choice, best accuracy/size trade-off |
| **Large** | ~1.9M | 384 | 12 | Maximum accuracy, server deployment |

---

## 📁 Project Structure

```markdown
MobilePlantViT-LDA/
├── src/
│   ├── __init__.py
│   ├── blocks/                    # Neural network building blocks
│   │   ├── __init__.py
│   │   ├── attention.py           # Linear Differential Attention
│   │   ├── classifier.py          # GAP and Classification Head
│   │   ├── coord_attention.py     # Coordinate Attention module
│   │   ├── ffn.py                 # Bottleneck Feed-Forward Network
│   │   ├── fused_ir.py            # Fused Inverted Residual Block
│   │   ├── ghost_conv.py          # Ghost Convolution
│   │   ├── patch_embed.py         # Patch Embedding
│   │   ├── positional_encoding.py # Sinusoidal Positional Encoding
│   │   └── utils.py               # Utility functions
│   └── models/
│       ├── __init__.py
│       └── mobile_plant_vit.py    # Main model implementation
├── preprocessing/
│   └── preprocessing_color.ipynb  # Data preprocessing pipeline
├── training/
│   └── training-color-mobileplantvit-large.ipynb  # Training notebook
└── README.md
```



---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/MobilePlantViT-LDA.git
cd MobilePlantViT-LDA
pip install torch torchvision numpy matplotlib pillow tqdm scikit-learn
```

<!-- need to add more -->

<!-- 👇last part added already -->
---

## 🔬 Technical Details

### Attention Mechanism

The Linear Differential Attention computes:

1. Project input to Q₁, Q₂, K₁, K₂, V
2. Compute attention scores: `A₁ = softmax(Q₁K₁ᵀ/√d)`, `A₂ = softmax(Q₂K₂ᵀ/√d)`
3. Differential attention: `A_diff = α × (A₁ - A₂)`
4. Apply to values: `output = A_diff × V`

### Parameter Efficiency

The architecture achieves parameter efficiency through:
- **GhostConv**: Generates features cheaply via depthwise operations
- **Bottleneck FFN**: Contracts then expands (opposite of standard transformer)
- **Single Transformer Block**: Minimal transformer overhead
- **Efficient Projections**: Careful dimensionality choices

---

## 📚 References

- [GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907)
- [Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907)
- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
- [DIFF Transformer](https://arxiv.org/abs/2410.05258)
- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## ⭐ Acknowledgments

- PlantVillage dataset for providing the plant disease images
- The PyTorch team for the excellent deep learning framework
- The research community for the foundational architectures and techniques