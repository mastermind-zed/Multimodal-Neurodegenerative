# Task: Multimodal Neurodegenerative Disease Research (AD & PD)

**Aim**: To improve early detection and progression prediction of AD and PD using multimodal fusion of MRI imaging and clinical markers, tailored for the Ghanaian healthcare context.

- [/] Data Acquisition & Preprocessing [x]
- [x] Download OASIS (AD) and Parkinson's MRI datasets
- [x] Clean and normalize clinical metadata (AD: [x], PD: [x])
- [x] Preprocess MRI images (AD: [x], PD: [x])
- [x] Implement data loaders for multimodal input

## Phase 2: Model Development [x]
- [x] Develop CNN backbones for structural MRI features
- [x] Implement MLP/Embedding layers for clinical metadata
- [x] Design and implement Multimodal Fusion Layers (Late/Hybrid Attention)
- [x] Train and evaluate unified/specific models for AD and PD (PD: [x], AD: [x - Ready])

## Phase 3: XAI & Fairness [x]
- [x] Integrate Grad-CAM for biomarker visualization (Hippocampus/Basal Ganglia)
- [x] Evaluate model fairness and bias for African population representativeness
- [ ] Assess lightweight deployment for mobile-aided diagnostics in Ghana

## Phase 4: Finalization [ ]
- [ ] Create comprehensive research walkthrough
- [ ] Export models to ONNX/TFLite
