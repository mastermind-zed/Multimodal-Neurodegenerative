# Multimodal Neurodegenerative Disease Research Implementation Plan

## Problem Statement
Early detection of Alzheimer’s (AD) and Parkinson’s (PD) is often hindered by the limitations of structural MRI alone (missing prodromal markers) and late clinical presentation in sub-Saharan Africa. Complementary clinical data is often omitted, leading to lower diagnostic sensitivity.

## Aim
To develop a unified multimodal fusion framework that integrates MRI imaging with clinical/demographic metadata for early detection and progression prediction of AD and PD.

## Proposed Strategy
- **Modality 1: Imaging**: High-resolution 2D/3D MRI processed via CNNs (ResNet/DenseNet).
- **Modality 2: Clinical**: MMSE, CDR, motor scores, Age, and Progression markers processed via MLPs.
- **Fusion**: Late Fusion (feature concatenation) or Hybrid Fusion (cross-attention) to capture joint biomarkers.

## Project Structure
- `data/alzheimer/` & `data/parkinsons/`
- `scripts/`: Shared and disease-specific preprocessing/training scripts.
- `models/`: Modular architecture for easy switching between AD and PD backbones.
- `metadata/`: Clinical datasets for both cohorts.

## Verification Plan
- Cross-validation on OASIS and Parkinson's cohorts.
- XAI validation: Focus on Hippocampus (AD) vs. Basal Ganglia (PD).
- Fairness audit across age and demographic subgroups.
