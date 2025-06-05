# DuoProto: Dual-Branch Prototype-Guided Framework for Early Recurrence Prediction in HCC

## Installation

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.8
- PyTorch >= 1.9.0
- MONAI >= 0.9.0
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy

## Data Preparation

### Data Format

Your data should be organized with the following structure:

**Single-phase data (PV only):**
```python
{
    "PVimg": "/path/to/pv_image.nii.gz",
    "PVmask": "/path/to/pv_mask.nii.gz", 
    "label": 0 or 1,  # ER (1) vs NER (0)
    "bclc": 0, 1, 2, or 3,  # BCLC staging (0, A, B, C)
    "PID": "patient_id"
}
```

**Multi-phase data:**
```python
{
    "preimg": "/path/to/pre_image.nii.gz",
    "premask": "/path/to/pre_mask.nii.gz",
    "Aimg": "/path/to/arterial_image.nii.gz", 
    "Amask": "/path/to/arterial_mask.nii.gz",
    "PVimg": "/path/to/pv_image.nii.gz",
    "PVmask": "/path/to/pv_mask.nii.gz",
    "Delayimg": "/path/to/delay_image.nii.gz",
    "Delaymask": "/path/to/delay_mask.nii.gz",
    "label": 0 or 1,  # ER (1) vs NER (0)
    "bclc": 0, 1, 2, or 3,  # BCLC staging (0, A, B, C)
    "PID": "patient_id"
}
```

### Preprocessing Pipeline

Following the methodology from our paper:

1. **Resampling**: All CT scans to uniform 1.0×1.0×1.0 mm³ voxel spacing
2. **Intensity Normalization**: Clip to [-21, 189] HU and normalize to [0, 1]
3. **Liver Segmentation**: Using SegVol pretrained model for automated liver ROI extraction
4. **Spatial Standardization**: Crop liver bounding box and pad/crop to 192×192×192
5. **Data Augmentation**: Affine transformations, contrast adjustment, Gaussian noise

### Custom Data Loading

Implement your data loading logic in `utils/dataloader.py` by modifying the `get_custom_data()` function:

```python
def get_custom_data():
    # TODO: Implement your custom data loading here
    files = []  # List of data dictionaries
    labels = []  # List of corresponding labels
    
    # Your data loading logic here
    # ...
    
    return files, labels
```

## Usage

### Training

Run training with the DuoProto framework:

```bash
bash train.sh
```

Or customize training parameters:

```bash
python main.py \
    --model="ReVit_avg" \
    --dataset="CUSTOM_DATASET" \
    --device="cuda:0" \
    --batch_size=8 \
    --epochs=150 \
    --num_classes=2 \
    --test_size=0.3 \
    --prototype_momentum=0.9 \
    --er_ce_lambda=1.0 \
    --proto_lambda=0.6 \
    --proto_sep_lambda=0.8 \
    --er_rank_lambda=0.5 \
    --align_lambda=0.1
```

### Key Training Parameters

- `--fusion_method`: Multi-phase feature fusion (`attention`, `concat`, `gated`)
- `--prototype_momentum`: EMA momentum for prototype updates (default: 0.9)
- `--er_ce_lambda`: Weight for classification loss (α = 1.0)
- `--proto_lambda`: Weight for contrastive prototype loss (β = 0.6)
- `--proto_sep_lambda`: Weight for prototype separation loss (γ = 0.8)
- `--er_rank_lambda`: Weight for BCLC ranking loss (δ = 0.5)
- `--align_lambda`: Weight for prototype alignment loss (λ = 0.1)



## Experimental Results

### Performance Comparison

DuoProto consistently outperforms baseline methods:

| Method | AUPRC | AUROC | F1 Score | Sensitivity | Precision |
|--------|-------|-------|----------|-------------|-----------|
| Radiomics | 0.5644 | 0.6560 | 0.6141 | 0.5358 | 0.4658 |
| ResNet10 | 0.5321 | 0.6521 | 0.5934 | 0.5471 | 0.4695 |
| ViT | 0.4758 | 0.6307 | 0.6355 | 0.5214 | 0.4526 |
| Swin Transformer | 0.5378 | 0.6457 | 0.5857 | 0.5853 | 0.4922 |
| ReViT | 0.5647 | 0.6657 | 0.5817 | 0.5786 | 0.5013 |
| **DuoProto (Ours)** | **0.6482** | **0.7438** | **0.6647** | **0.6674** | **0.5305** |

### Ablation Study

| Component Removed | AUPRC | AUROC | Impact |
|------------------|-------|-------|---------|
| w/o L_proto | 0.5888 | 0.6564 | -9.1% AUPRC |
| w/o L_sep | 0.5774 | 0.6819 | -10.9% AUPRC |
| w/o L_rank | 0.5829 | 0.6799 | -10.0% AUPRC |
| w/o L_align | 0.6040 | 0.7113 | -6.8% AUPRC |
| **Full Model** | **0.6482** | **0.7438** | **Best** |


## File Structure

```
├── main.py                 # Main training script
├── train.sh               # Training shell script  
├── requirements.txt       # Python dependencies
├── utils/
│   ├── dataloader.py     # Data loading utilities
│   └── balanced_sampler.py # Balanced sampling for class imbalance
├── models/
│   ├── ViT.py            # Vision Transformer implementation
│   ├── multiphase_vit.py # Multi-phase ViT model (auxiliary branch)
│   └── proto_model.py    # DuoProto fusion model
├── trainer/
│   └── training.py       # Dual-branch training loop
└── inference/
    └── evaluation.py     # Evaluation metrics and visualization
```

## Implementation Details

### Training Configuration

- **Hardware**: NVIDIA A100 GPU (80GB VRAM)
- **Batch Size**: 8 (balanced sampling for ER/NER)
- **Epochs**: 150 with early stopping
- **Data Split**: 60% train, 10% validation, 30% test
- **Optimizers**: Separate AdamW for each branch
  - Multi-phase branch: 5e-4 learning rate
  - Single-phase branch: 3e-4 learning rate
- **Warm-up**: Linear warm-up for multi-phase branch (10 epochs)

### Loss Function Weights

Based on empirical validation in our paper:
- α = 1.0 (Classification loss)
- β = 0.6 (Prototype contrastive loss) 
- γ = 0.8 (Prototype separation loss)
- δ = 0.5 (BCLC ranking loss)
- λ = 0.1 (Prototype alignment loss)

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{yu2024duoproto,
  title={Learning from Limited Multi-Phase CT: Dual-Branch Prototype-Guided Framework for Early Recurrence Prediction in HCC},
  author={Yu, Hsin-Pei and Lyu, Si-Qin and Hsieh, Yi-Hsien and Wang, Weichung and Su, Tung-Hung and Kao, Jia-Horng and Lin, Che},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE}
}
```

## Ethics Statement

The dataset used in this study was approved by the Institutional Review Board of National Taiwan University Hospital (IRB No. 202306004RINC). All data collection and analysis procedures complied with relevant ethical guidelines and regulations.

## Acknowledgments

This work was sponsored by:
- National Science and Technology Council (NSTC, 113-2222-E-002-008)
- Ministry of Health and Welfare (MOHW, 114-TDU-B-221-144003)  
- Ministry of Education (MOE, 113M7054) in Taiwan
- Center for Advanced Computing and Imaging in Biomedicine (NTU-114L900701)
