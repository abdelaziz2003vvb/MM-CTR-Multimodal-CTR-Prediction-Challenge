# WWW2025 MMCTR Challenge - Tasks 1 & 2 Solution

[![WWW 2025](https://img.shields.io/badge/WWW-2025-blue)](https://www2025.thewebconf.org/)
[![FReL@MIR Workshop](https://img.shields.io/badge/Workshop-FReL%40MIR-green)](https://www.codabench.org/competitions/5372/)
[![License](https://img.shields.io/badge/license-Academic-orange)]()

## Overview

This repository contains a comprehensive solution for the **WWW2025 Multimodal CTR Prediction Challenge** hosted at the 1st FReL@MIR Workshop. The challenge is part of the Multimodal Information Retrieval Challenge (MIRC) track at The Web Conference 2025 in Sydney, Australia.

### Challenge Goals

The Multimodal Click-through Rate (MM-CTR) Challenge focuses on two fundamental tasks in multimodal recommendation systems:

1. **Task 1: Multimodal Item Embedding** - Develop robust multimodal item representation learning methods
2. **Task 2: Multimodal CTR Prediction** - Design effective CTR prediction models leveraging multimodal embeddings

### Dataset

We use the **MicroLens-1M dataset** released by Westlake University, containing:
- **1M users** and **91K items**
- **Multimodal content**: Cover images, videos, titles, and detailed item features
- **Public data** for training and **private data** for testing

## Solution Architecture

### Task 1: Multimodal Item Embedding

**Objective:** Extract and fuse multimodal features to generate high-quality 128-dimensional item embeddings.

#### Challenge Requirements:

**Input:**
- User historical interaction data
- Item multimodal data (attributes, title, cover images)

**Output:**
- Item embeddings (dimension size: 128)

**Evaluation:**
- Participants must replace embedding features in the provided CTR model
- Retrain the model and evaluate on validation/test sets
- Metric: **AUC (Area Under Curve)**
- Final leaderboard based on reproducible results

**Constraints:**
- ✅ Using any data outside the provided dataset is **not allowed**
- ✅ Large models can be used to extract features if open-source
- ✅ Using user historical data for model training is allowed (but cannot use downstream CTR model directly)

#### Our Pipeline:

1. **Image Feature Extraction**
   - Model: CLIP-ViT-Large-Patch14 (`openai/clip-vit-large-patch14`)
   - Output: 768-dimensional image embeddings
   - Processing: L2 normalization for all features

2. **Text Feature Extraction**
   - Model: Multilingual-E5-Base (`intfloat/multilingual-e5-base`)
   - Input: Item titles with attention-mask-aware mean pooling
   - Output: 768-dimensional text embeddings
   - Processing: L2 normalization

3. **Multi-Modal Fusion**
   - Concatenation: Image (768D) + Text (768D) = 1536D
   - Dimensionality Reduction: PCA to 128D
   - Variance Retained: ~60.4%
   - Final Processing: L2 normalization

4. **Integration**
   - Embeddings properly aligned with `item_id` indices
   - Padding token (ID=0) set to zero vector
   - Saved as frozen features for Task 2

### Task 2: Multimodal CTR Prediction

**Objective:** Leverage multimodal embeddings to build effective CTR prediction models.

#### Challenge Requirements:

**Input:**
- User click data
- All items' multimodal embeddings (from Task 1 or provided baseline)

**Output:**
- Predicted pCTR for test set

**Evaluation:**
- Replace baseline model with customized CTR model from Task 2
- Train and evaluate on validation set, predict on test set
- Metric: **AUC (Area Under Curve)**
- Final leaderboard based on reproducible results

**Constraints:**
- ❌ Using pretrained large models directly for CTR prediction is **not allowed**
- ✅ Can be used for item embedding extraction in Task 1
- ✅ The designed CTR model must be trained **end-to-end** with given features
- ✅ For practical considerations, only single model using neural networks is allowed
- ❌ Ensemble methods or tree models are **not allowed** for training or testing

#### Our Architecture:

```
Input Layer
├── Sequence Encoding (item_seq)
│   ├── Trainable ID Embeddings (64D)
│   └── Frozen MM Embeddings (128D from Task 1)
│
├── Target Item Encoding
│   ├── Trainable ID Embeddings (64D)
│   └── Frozen MM Embeddings (128D)
│
└── Context Features
    ├── Likes Level Embedding (16D)
    └── Views Level Embedding (16D)

↓

SASRec Transformer Block
├── Positional Encoding
├── 2 Transformer Encoder Layers
├── 4 Attention Heads
├── FFN Hidden Size: 512
└── Output: User Vector (192D)

↓

Feature Concatenation (192D user + 192D target + 32D context = 416D)

↓

DCN-v2 (Deep & Cross Network v2)
├── Cross Network (3 layers)
│   └── Explicit feature interactions
└── Deep Network (512 → 256 → 128)
    └── Implicit feature learning

↓

Output Layer (Binary Classification)
└── BCEWithLogitsLoss
```

#### Key Features:

- **Hybrid Embeddings:** Combines trainable ID embeddings with frozen multi-modal features
- **Sequential Modeling:** SASRec captures temporal patterns in user behavior
- **Feature Interactions:** DCN-v2 models both low and high-order interactions
- **Training Techniques:**
  - Mixed precision training (AMP)
  - Gradient clipping (max_norm=1.0)
  - Early stopping (patience=3)
  - ReduceLROnPlateau scheduler

## Results

### Training Performance:

- **Best Validation AUC:** 0.8909
- **Training Strategy:** 4 epochs (early stopped)
- **Model Parameters:** ~7.46M (all trainable)
- **Optimization:** AdamW with initial LR=1e-3

### Dataset Statistics:

- **Training Set:** 3.6M interactions
- **Validation Set:** 10K interactions
- **Test Set:** 379K interactions
- **Vocabulary Size:** 91,705 items

## Installation

### Requirements:

```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install scikit-learn
pip install pandas polars
pip install pillow
pip install tqdm
pip install joblib
pip install matplotlib
```

### Hardware Requirements:

- **GPU:** NVIDIA Tesla T4 or better (16GB+ VRAM recommended)
- **RAM:** 32GB+ recommended for embedding generation
- **Storage:** ~50GB for data + models + embeddings

## Usage

### Running the Complete Pipeline:

```bash
jupyter notebook www2025-mmctr-challenge-task1-2.ipynb
```

Or execute cells sequentially:

1. **Cell 1-3:** Setup and imports
2. **Cell 4-5:** Load and verify data
3. **Cell 6-9:** Generate Task 1 embeddings
4. **Cell 10:** Align embeddings with item IDs
5. **Cell 11-13:** Load embeddings and prepare data
6. **Cell 14-16:** Build Task 2 model
7. **Cell 17-18:** Train model
8. **Cell 19:** Visualize training curves
9. **Cell 20:** Generate submission

### Directory Structure:

```
/kaggle/
├── input/
│   └── www2025-mmctr-data/
│       └── MicroLens_1M_MMCTR/
│           ├── item_images/          # 91,717 images
│           ├── item_feature.parquet
│           ├── item_seq.parquet
│           └── MicroLens_1M_x1/
│               ├── train.parquet
│               ├── valid.parquet
│               ├── test.parquet
│               └── item_info.parquet
└── working/
    └── mmctr_task1_v2/
        ├── item_img_emb_clipL_768.npy      # Image embeddings
        ├── item_txt_emb_e5_768.npy         # Text embeddings
        ├── item_mm_emb_128.npy             # Final MM embeddings
        ├── pca_mm_128.joblib               # PCA model
        ├── item_info.parquet               # Updated with embeddings
        ├── best_model.pt                    # Best checkpoint
        ├── training_curves.png              # Training visualization
        └── prediction.csv                   # Submission file
```

## Configuration

Key hyperparameters defined in `Paths` class:

```python
# Model Architecture
MAX_SEQ_LEN = 50
D_ID_EMB = 64          # Trainable item ID embedding
D_MM_EMB = 128         # Frozen multimodal embedding
NUM_LAYERS = 2         # Transformer layers
NUM_HEADS = 4          # Attention heads
NUM_CROSS = 3          # DCN cross layers

# Training
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-6
EPOCHS = 20
PATIENCE = 3
GRAD_CLIP = 1.0
```

## Submission Types

The challenge offers three submission tracks:

### Track 1: Task 1 Only
- **Focus:** Optimize embeddings with **fixed CTR model**
- **Constraint:** Cannot modify the baseline CTR model code
- **Flexibility:** Can adjust embedding hyperparameters

### Track 2: Task 2 Only  
- **Focus:** Optimize CTR model with **fixed embeddings**
- **Constraint:** Cannot change input features (use provided embeddings)
- **Flexibility:** Can optimize model structure and hyperparameters

### Track 3: Tasks 1 & 2 (End-to-End) ⭐ **Our Approach**
- **Focus:** Cascaded optimization of both tasks
- **Flexibility:** Can modify both embeddings and CTR model
- **Constraint:** Two separate models required for Task 1 and Task 2
- **Benefit:** Better optimization of item embedding features and CTR modeling

> 💡 **Our Solution:** We chose Track 3 to fully enhance the multimodal recommendation pipeline and design better solutions for both embedding extraction and CTR prediction.

---

## Dataset Description

### MicroLens-1M Dataset

- **Source:** Westlake University
- **Scale:** 1M users, 91K items
- **Download:** [https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR/](https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR/)

### Data Samples

The dataset includes rich multimodal information:

| Cover | Video | Title | Likes | Views |
|-------|-------|-------|-------|-------|
| 🍜 Cooking | 📹 Recipe Demo | "Spicy Ruff Demon-king Stir-Fried Nooddles Challenge!" | 41M | 1.2M |
| 🐱 Cute Cat | 📹 Pet Video | "Turns out even little kittens can have emo moments..." | 26K | 1.23M |
| 🐶 Dog Playing | 📹 Outdoor Fun | "Can you catch fish by soaking your paws in the countryside" | 10K | 364K |
| 🐕 Beach Dog | 📹 Beach Scene | "What kind of figurine are you? Dalmatian fight club" | 25K | 423K |
| 🏙️ City View | 📹 Travel Vlog | "What the four Seasons Scenery in Estonia Looks Comes to Life..." | 318K | 1.13M |

---

### 1. SASRec (Self-Attentive Sequential Recommendation)
- Captures long-term dependencies in user behavior
- Positional encoding for sequence order
- Multi-head self-attention
- Layer normalization

### 2. DCN-v2 (Deep & Cross Network v2)
- **Cross Network:** Explicit feature crossing
- **Deep Network:** Implicit feature learning
- Parallel architecture for complementary learning

### 3. Embedding Strategy
- **Trainable ID Embeddings:** Learn item-specific patterns
- **Frozen MM Embeddings:** Leverage pre-trained visual-semantic knowledge
- **Concatenation:** Best of both worlds

## Participating

### Official Challenge Portal

🔗 **CodaBench:** [https://www.codabench.org/competitions/5372/](https://www.codabench.org/competitions/5372/)

### Important Information

- **Workshop:** 1st FReL@MIR Workshop
- **Conference:** The Web Conference 2025 (WWW 2025)
- **Location:** Sydney, Australia
- **Challenge Goal:** Advance multimodal recommendation with practical solutions for industrial recommender systems

---

### Common Issues:

1. **CUDA Out of Memory:**
   - Reduce `BATCH_SIZE` in configuration
   - Use gradient checkpointing
   - Close other GPU processes

2. **Slow Embedding Generation:**
   - Adjust `BATCH_SIZE_IMG` and `BATCH_SIZE_TXT`
   - Reduce `NUM_WORKERS`

3. **PCA Variance Low:**
   - Check image/text embedding quality
   - Verify L2 normalization
   - Consider alternative fusion methods

## Citation

If you use this code or find it helpful, please cite:

```bibtex
@inproceedings{www2025mmctr,
  title={Multimodal CTR Prediction Challenge},
  booktitle={The 1st FReL@MIR Workshop at WWW 2025},
  year={2025},
  address={Sydney, Australia}
}
```

## License

This project is for academic and research purposes. Please refer to the WWW2025 MMCTR Challenge guidelines for usage terms.

## Acknowledgments

- **WWW 2025 MMCTR Challenge Organizers** for hosting this competition
- **Westlake University** for the MicroLens-1M dataset
- **OpenAI** for CLIP visual encoders
- **Microsoft** for E5 text embeddings
- **Kaggle** for computational resources and platform
- **FReL@MIR Workshop** organizers

---

## Contact & Support

- **Challenge Portal:** [CodaBench Competition Page](https://www.codabench.org/competitions/5372/)
- **Issues:** Open an issue in this repository
- **Dataset:** [MicroLens-1M Download](https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR/)

---

## Competition Details

### Timeline
- Check the official CodaBench page for up-to-date deadlines

### Leaderboard
- **Preliminary Round:** pCTR predictions only
- **Final Round:** Submit reproducible training code + inference code
- **Ranking:** Based on AUC metric with reproducibility verification

### Winner Announcements
- Results will be publicly announced on the official website
- Top solutions may be invited to present at the FReL@MIR Workshop

---

**Note:** This solution achieved a validation AUC of **0.8909** in our experiments. Actual test performance may vary based on the private test set distribution.
