# From Accuracy to Reliability: Calibrating Deep Neural Networks for Chest X-Ray Classification

## Master's Thesis Project
**Author:** Mohamed Souleimane Cheikh Ahmed  
**Institution:** Istinye University, Institute of Graduate Education  
**Advisor:** Assoc. Prof. Dr. Mustafa Sundu  
**Year:** 2026  

## Project Overview
Deep learning models in medical imaging often achieve high accuracy but fail to provide reliable probability estimates—a critical flaw in clinical decision-making. A model that predicts a disease with **90% confidence** should be correct **90% of the time**. However, modern neural networks are often **miscalibrated**, tending to be overconfident in their incorrect predictions.

This research addresses this gap by implementing and evaluating **post-hoc calibration techniques** on the **CheXpert** dataset. The goal is not just to classify 5 thoracic pathologies (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion) but to ensure the model's predicted probabilities are trustworthy.

## The Dataset: CheXpert
- **Source:** Stanford ML Group
- **Size:** 224,316 Chest X-Rays.
- **Pathologies Targeted:**
  1.  Atelectasis
  2.  Cardiomegaly
  3.  Consolidation
  4.  Edema
  5.  Pleural Effusion
- **Label Handling:** The "U-Ones" policy was used, treating uncertain labels as positive to prioritize sensitivity.

## Methodology & Architecture

### 1. Base Model: DenseNet121
I utilized **DenseNet121**, pretrained on ImageNet, as the backbone architecture.
- **Why DenseNet?** Its dense connectivity pattern improves feature propagation and reduces the number of parameters compared to ResNet, making it highly effective for medical image feature extraction.
- **Training Setup:**
  - **Optimizer:** Adam
  - **Loss Function:** Binary Cross-Entropy (BCE)
  - **Image Resolution:** 320x320
  - **Data Augmentation:** Random rotation, horizontal flipping, contrast adjustments.

### 2. Calibration Techniques Implemented
To fix the overconfidence of the base model, I implemented three post-hoc calibration methods:
1.  **Temperature Scaling:** A parametric method that scales the logits by a single scalar parameter $T$ (temperature) to soften the output distribution.
2.  **Isotonic Regression:** A non-parametric approach that fits a piecewise constant function to the uncalibrated probabilities.
3.  **Platt Scaling:** Logistic regression applied to the model's output scores.

### 3. Evaluation Metric: Expected Calibration Error (ECE)
Unlike standard projects that only report AUROC, this research optimized for **ECE (Expected Calibration Error)**, which measures the weighted average difference between predicted confidence and actual accuracy.

## Key Results
The study demonstrated that while the base DenseNet121 model achieved high AUROC, it was significantly miscalibrated. **Temperature Scaling** proved to be the most effective method for improving reliability without sacrificing discrimination performance.

<img width="718" height="154" alt="Screenshot 2026-01-19 at 00 28 50" src="https://github.com/user-attachments/assets/ffeeb443-b92f-4f57-9ef3-e60745988d27" />


## Pwe label AUC
<img width="2552" height="1223" alt="per_label_auc_heatmap_clean" src="https://github.com/user-attachments/assets/f06cf554-8897-47fb-b32f-8b2a26d7dc01" />


### Visualizing Calibration
The reliability diagrams below show the shift from an overconfident model (Uncalibrated) to a perfectly aligned one (Calibrated).

<img width="719" height="531" alt="Screenshot 2026-01-19 at 00 31 08" src="https://github.com/user-attachments/assets/8c85bdfa-1ba6-414b-8894-6aa6d9970dc4" />
