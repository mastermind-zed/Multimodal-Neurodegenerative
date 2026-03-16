# Multimodal Neurodegenerative Research Walkthrough

This document summarizes the initialization and verification of the unified project for early detection of Alzheimer’s and Parkinson’s using multimodal fusion.

## Project Architecture
We have implemented a **Late Fusion** architecture that combines:
- **Imaging Branch**: ResNet-18 backbone for structural MRI features.
- **Clinical Branch**: Multi-Layer Perceptron (MLP) for demographic and cognitive markers (Age, MMSE, CDR, UPDRS).
- **Fusion Head**: Concatenates features for joint disease classification.

## Key Accomplishments

### 1. Data Acquisition & Preprocessing
- **Merged Repository**: Both AD and PD studies are now unified in `Multimodal Neurodegenerative Research`.
- **MRI Processing**: 86,442 AD images and 831 PD images have been normalized and resized to 224x224.
- **Clinical Simulation**: Since the raw datasets were image-only, we generated **87,000+ realistic clinical records** (MMSE, CDR, Age) in `metadata/` to enable multimodal development.

### 2. Implementation
- **Modular Data Loader**: `MultimodalDataset` handles synchronous ingestion of images and tabular data.
- **Fusion Model**: `FusionModel` implemented in `models/fusion_model.py`.
- **Explainability**: `explainability.py` provides a Grad-CAM framework to visualize the "why" behind the model's neuro-imaging predictions.

### 3. Verification Results (Parkinson's Test Case)
- **Training**: Verified through a 3-epoch test run on the PD branch.
- **Metrics**: Successfully generated a classification report and confusion matrix in `results/`.
- **Loss Progression**: Smooth convergence observed (Final Loss: 0.0197).

## How to Start Full Training

### For Alzheimer's (OASIS):
```bash
python scripts/train_multimodal.py
```
*(Note: I set the default in the script to Parkinson's for testing; you can change the `__main__` disease type to "alzheimer" for the large-scale run.)*

### For Evaluation:
```bash
python scripts/evaluate_multimodal.py
```

## Next Steps for PhD Research
- **Refinement**: Replace simulated clinical data with real-world Ghanaian patient metadata if available.
- **Explainability**: Run Grad-CAM on AD results to confirm focus on hippocampal atrophy regions.
- **Bias Mitigation**: Evaluate the model's sensitivity across different age tiers.
