README: Transformer-Based Cyberbullying Detection: A RoBERTa Fine-Tuning Approach for Spanish Social Media

1. Reproducibility
This repository contains all scripts required to train, validate, and evaluate a RoBERTa-based model for cyberbullying detection in Spanish-language social media posts using stratified 5-fold cross-validation.
Full model checkpoints, per-epoch logs, and configuration files are saved for each fold. The training pipeline is deterministic with manual seeds for reproducibility.
2. Title
Transformer-Based Cyberbullying Detection: A RoBERTa Fine-Tuning Approach for Spanish Social Media3. Description
This project focuses on detecting cyberbullying in Spanish social media using a transformer-based (RoBERTa) model. It addresses:


+ Class imbalance via data augmentation and weighted loss
+ Generalization using stratified K-fold cross-validation
+Model robustness through early stopping and macro-averaged metrics (F1, accuracy)
4. Dataset Information


Name: augmented_dataset.csv
Source: Hugging Face dataset: manueltonneau/spanish-hate-speech-superset
Languages: Spanish (Iberian and Latin American)
Classes:
0: No Cyberbullying
1: Cyberbullying
Distribution:
Before augmentation: 19,438 (no), 4,947 (yes)
After augmentation: 22,590 (no), 9,444 (yes)
Total Samples: 32,034
5. Code Information
Language: Python 3.9+
Key Libraries:
transformers, datasets (Hugging Face)
scikit-learn (metrics and k-fold splitting)
pandas, numpy, torch
Inputs:
CSV file with columns: text, labels
Tokenized dataset saved as Hugging Face Dataset object
Outputs:
Model checkpoints per fold
Logs of validation accuracy/loss/F1 for each epoch per fold
Final aggregated performance metrics
 
6. Usage Instructions
Install dependencies:
bash
pip install -r requirements.txt
Prepare dataset:
Ensure augmented_dataset.csv exists with text and labels columns.
(Optional) Tokenize and save using Hugging Face Datasets:
python
tokenized_dataset.save_to_disk("tokenized_dataset")
Run training script:
bash
python run_cyberbullying_cv.py
Outputs:
Logs: epoch_log_foldX.txt
Models: best_model_foldX/
Final metrics printed to console
7. Requirements
Python 3.8 or higher
PyTorch ≥ 1.13
Hugging Face Transformers ≥ 4.12
scikit-learn
pandas, numpy
CUDA (if using GPU)
8. Methodology
Model: PlanTL-GOB-ES/roberta-base-bne, fine-tuned with a binary classifier head
Splitting: 5-fold stratified cross-validation
Training:
Optimizer: Adam
Weighted loss: class weights computed via compute_class_weight
Early stopping: patience = 10
Metrics:
Validation Accuracy
Macro F1-Score
Loss Function: Custom weighted cross-entropy
9. Citations (If Applicable)
Please cite the following if you use this code:
Hugging Face Transformers:
Wolf et al. Transformers: State-of-the-Art Natural Language Processing (2020).
Spanish RoBERTa:
PlanTL-GOB-ES/roberta-base-bne on Hugging Face
If using the spanish-hate-speech-superset dataset:
text
@misc{manueltonneau2021dataset,
  author = {Manuel Tonneau},
  title = {Spanish Hate Speech Superset},
  year = {2021},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/manueltonneau/spanish-hate-speech-superset}}
}
10. License & Contribution Guidelines
License: This project uses datasets and models under Apache 2.0 and CC BY-SA 4.0 (see individual Hugging Face model cards for details). Code is open for educational and research use.
Contributing: If submitting issues or feature requests, open a GitHub Issue or PR using clear documentation.
11. Materials & Methods
OS: Ubuntu 20.04 / Windows 10
Hardware: NVIDIA RTX 3080 / 3090 (8–24GB VRAM recommended), or Google Colab Pro
Environment:
Python 3.9
CUDA 11.8 (if using GPU)
Training Time: 2–4 hours per fold on a single GPU
12. Conclusions
Generalization: The model demonstrates consistent macro F1/accuracy across folds.
Data Augmentation: Class augmentation significantly improved balance and learning.
Bias Awareness: Explicit mention of possible cultural and annotation bias.
Modularity: The system is modular, supporting hyperparameter optimization and model upgrades (e.g., multilingual RoBERTa).
Limitations
Bias: Annotator and dialect biases in the original dataset.
Generalizability: Model trained on Twitter/Reddit; may not generalize to WhatsApp or TikTok.
Explainability: While effective, model decisions are not yet interpretable in a human-readable sense.
Fold 3 Exception: If excluded (due to logging error), document as noted in manuscript.
