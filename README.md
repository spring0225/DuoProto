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
bash scripts/train.sh
```

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
