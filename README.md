# Project 11: Fine-tuning Pre-trained ResNet50 for Custom Image Classification

Fine-tuning a pre-trained ResNet50 CNN on the TensorFlow Flower Photos dataset (5 classes) using a two-phase transfer learning strategy — first training a custom classification head with the backbone frozen, then selectively unfreezing the top layers for fine-tuning.

---

## Dataset

**TensorFlow Flower Photos** — downloaded directly from Google Storage at runtime.

| Class | Description |
|-------|-------------|
| daisy | ~633 images |
| dandelion | ~898 images |
| roses | ~641 images |
| sunflowers | ~699 images |
| tulips | ~799 images |

Total: ~3,670 images across 5 classes. Split 80/20 into training and validation sets via `ImageDataGenerator(validation_split=0.2)`.

---

## Model Architecture

**Base:** ResNet50 pre-trained on ImageNet (`include_top=False`, `input_shape=(224, 224, 3)`)

**Custom Classification Head:**
```
GlobalAveragePooling2D
Dense(64, activation='relu')
BatchNormalization
Dropout(0.5)
Dense(32, activation='relu')
Dropout(0.3)
Dense(5, activation='softmax')
```

---

## Training Strategy

### Phase 1 — Transfer Learning (Frozen Backbone)
- ResNet50 base: fully frozen
- Only the custom head is trained
- Optimizer: Adam — LR 0.001
- Epochs: up to 25 (EarlyStopping on val_accuracy, patience=7)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Phase 2 — Fine-tuning (Partial Unfreeze)
- First 100 layers of ResNet50 remain frozen
- Top layers (layer 100 onward) unfrozen for fine-tuning
- Optimizer: Adam — LR 0.0001 (10× lower to protect pre-trained weights)
- Epochs: up to 20 (EarlyStopping on val_loss, patience=5)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## Data Augmentation

Training generator augmentation parameters:

| Transform | Value |
|-----------|-------|
| Rescale | 1/255 |
| Rotation range | 20° |
| Width / Height shift | 0.2 |
| Shear range | 0.2 |
| Zoom range | 0.2 |
| Horizontal flip | True |
| Fill mode | nearest |

Validation generator: rescale only (no augmentation).

---

## Evaluation

- Confusion matrix (seaborn heatmap)
- Per-class precision, recall, F1-score (`sklearn.metrics.classification_report`)
- Per-class accuracy from confusion matrix diagonal
- Visual prediction grid (10 samples, green = correct, red = incorrect)

---

## Key Findings

- A 64-unit dense head is optimal for a 5-class problem — larger heads overfit at this dataset size.
- Batch Normalization after the first dense layer improves training stability significantly.
- Reducing LR to 0.0001 in Phase 2 is critical — higher values destroy pre-trained feature representations.
- Conservative unfreezing (only layers beyond index 100) prevents catastrophic forgetting while still allowing domain adaptation.
- Fine-tuning consistently improves validation accuracy over head-only training.

---

## Repository Structure

```
├── ML_Task_11_Tayyabah_Rehman.ipynb   # Main notebook
├── resnet50_phase1_best.h5            # Best Phase 1 checkpoint
├── best_model_finetuned.h5            # Best Phase 2 checkpoint
├── phase1_results.png                 # Phase 1 training curves
└── README.md
```

---

## Requirements

```
tensorflow>=2.x
numpy
matplotlib
seaborn
scikit-learn
```

---

## How to Run

1. Open `ML_Task_11_Tayyabah_Rehman.ipynb` in Jupyter or Google Colab.
2. Run all cells in order — the dataset downloads automatically from Google Storage.
3. Phase 1 trains the custom head; Phase 2 fine-tunes the upper ResNet50 layers.
4. Final evaluation prints per-class metrics and renders the confusion matrix.

---

## Author

**Tayyabah Rehman**  
MPhil Artificial Intelligence — University of the Punjab, Lahore  
GitHub: [github.com/Tayyabah-Rehman](https://github.com/Tayyabah-Rehman)
