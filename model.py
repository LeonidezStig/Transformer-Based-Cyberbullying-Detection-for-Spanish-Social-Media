# Purpose: Train and evaluate a RoBERTa-based model for Spanish cyberbullying detection using stratified 5-fold cross-validation.
# Method: Tokenization with Hugging Face Transformers, cross-validation using StratifiedKFold, and performance metrics logging.
# Last updated: 2025-06-28
# Credit: Based on Hugging Face Transformers and Datasets libraries.

# PyTorch for building and training neural networks
import torch
# NumPy for numerical operations and array handling
import numpy as np
# pandas for reading and manipulating tabular datasets
import pandas as pd
# Hugging Face datasets for managing data in training and validation formats
from datasets import Dataset, DatasetDict
# Stratified K-Fold for balanced cross-validation
from sklearn.model_selection import StratifiedKFold
# Scikit-learn metrics for model evaluation
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, roc_curve, auc
# Matplotlib for plotting figures (e.g., ROC, confusion matrix)
import matplotlib.pyplot as plt
# Hugging Face transformers for tokenizer, model, trainer, and training configurations
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
# Early stopping to prevent overfitting if validation metric stops improving
from transformers import EarlyStoppingCallback
# Standard libraries for reproducibility and statistics
import random
from collections import Counter
# Seaborn for improved data visualizations
import seaborn as sns
# OS module for handling file system operations
import os

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the augmented dataset containing text and labels
df = pd.read_csv("augmented_dataset.csv")

# Print column names for verification
print("Dataset Columns:", df.columns)

# Specify the label column to be used for training
label_column = "labels"  # Change this if needed

from datasets import load_from_disk
# Load tokenized dataset that was previously saved using prepare_tokenized_data.py
tokenized_datasets = load_from_disk("tokenized_dataset")

# Load Spanish pre-trained RoBERTa model and tokenizer
model_name = "PlanTL-GOB-ES/roberta-base-bne"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize function for truncating and padding input text
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Set up 5-fold stratified cross-validation for balanced label distribution
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics for all folds
all_fold_metrics = []

all_val_preds = []
all_val_labels = []

full_dataset = tokenized_datasets  # It's already a Dataset object, not a dict
labels_array = np.array(full_dataset[label_column])
# Loop over each fold to perform training and evaluation
for fold, (train_index, val_index) in enumerate(skf.split(full_dataset, labels_array), 1):
    print(f"\nStarting fold {fold}...")

    # Split tokenized dataset into training and validation subsets for the current fold
    train_dataset = full_dataset.select(train_index)
    val_dataset = full_dataset.select(val_index)
    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

    # Load the model with a classification head for 2 classes
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Compute class weights to handle class imbalance during training
    from sklearn.utils.class_weight import compute_class_weight
    train_labels = np.array([example[label_column] for example in train_dataset])
    class_weights_np = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)

    # Define custom weighted cross-entropy loss function using computed class weights
    import torch.nn as nn
    class WeightedCrossEntropyLoss(nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.weights = weights
            self.ce = nn.CrossEntropyLoss(weight=self.weights)
        def forward(self, logits, labels):
            labels = labels.long()  # Ensure correct dtype
            return self.ce(logits, labels)
    loss_fn = WeightedCrossEntropyLoss(class_weights)

    epoch_logs = []

    from transformers import TrainerCallback

    # Custom callback to compute and store train accuracy after each epoch
    class TrainEvalCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
            trainer = kwargs.get("trainer", None)
            if trainer is not None:
                train_metrics = trainer.evaluate(trainer.train_dataset)
                trainer.state.log_history[-1]['train_accuracy'] = train_metrics.get("eval_accuracy", 0)
            return control

    # Metric computation function for accuracy and macro F1 score
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        return {"accuracy": acc, "f1": f1}
    
    # Log per-epoch results in a readable format including accuracy, loss, and F1
    def custom_log_callback(trainer):
        history = trainer.state.log_history
        seen_epochs = set()
        for entry in history:
            epoch = entry.get('epoch')
            if epoch is not None and int(epoch) == epoch and int(epoch) not in seen_epochs:
                seen_epochs.add(int(epoch))
                train_acc = entry.get('train_accuracy', entry.get('accuracy', 0))
                line = f"Epoch {int(epoch)}, Train Loss: {entry.get('loss', 0):.4f}, Val Loss: {entry.get('eval_loss', 0):.4f}, Val F1: {entry.get('eval_f1', 0):.4f}, Train Acc: {train_acc:.4f}"
                epoch_logs.append(line)

    # Override Trainer to use custom weighted loss during training
    from transformers import Trainer
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Define training arguments for batch size, learning rate, early stopping, etc.
    training_args = TrainingArguments(
        output_dir=f"./results_fold{fold}",
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=50,
        weight_decay=0.01,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        save_total_limit=1,
        seed=42,
        logging_dir=f"./logs_fold{fold}",
        logging_steps=50,
        report_to="none",
    )

    # Apply early stopping if validation metric doesn't improve for 10 epochs
    early_stopping = EarlyStoppingCallback(early_stopping_patience=10)

    # Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[early_stopping, TrainEvalCallback()]
    )

    checkpoint_path = f"./results_fold{fold}/checkpoint-last"
    # Resume training if a checkpoint exists; otherwise start fresh
    if os.path.exists(checkpoint_path):
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()
    # Log and save epoch-wise training progress for the current fold
    custom_log_callback(trainer)
    with open(f"epoch_log_fold{fold}.txt", "w") as f:
        for line in epoch_logs:
            f.write(line + "\n")

    # Display epoch logs in Colab for easy copy-paste
    print(f"\nEpoch Logs for Fold {fold}:")
    for line in epoch_logs:
        print(line)

    # Evaluate the model on the validation split after training
    eval_results = trainer.evaluate()
    print(f"Fold {fold} evaluation results:", eval_results)

    # Save the best model checkpoint for this fold
    trainer.save_model(f"best_model_fold{fold}")

    # Generate predictions on validation set to compute additional metrics and logs
    val_predictions_output = trainer.predict(dataset_dict["validation"])
    val_logits = val_predictions_output.predictions
    val_labels = val_predictions_output.label_ids
    val_preds = np.argmax(val_logits, axis=1)

    all_val_preds.extend(val_preds)
    all_val_labels.extend(val_labels)

    # Analyze and print class distribution of predictions
    pred_counts = Counter(val_preds)
    print(f"Fold {fold} - Prediction Distribution:", pred_counts)


    # Store accuracy and F1 metrics for this fold
    all_fold_metrics.append({
        "fold": fold,
        "accuracy": eval_results['eval_accuracy'],
        "f1": eval_results['eval_f1'],
        "roc_auc": None  # Removed ROC AUC calculation
    })

# Compute and display average accuracy and F1 score across all folds
avg_accuracy = np.mean([m["accuracy"] for m in all_fold_metrics])
avg_f1 = np.mean([m["f1"] for m in all_fold_metrics])

print("\n\n===== Combined Results Across All Folds =====")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")