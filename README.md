This project implements a transformer-based model using RoBERTa-bne for the automatic detection of cyberbullying in Spanish-language social media content. It includes a series of Python scripts and data files that collectively prepare, tokenize, train, and evaluate the model. The model.py script serves as the core component, orchestrating a 5-fold stratified cross-validation training loop, applying Hugging Face’s Trainer API, and saving detailed metrics at each epoch. Supporting scripts include prepare_dataset.py, which filters and loads the original dataset (es_hf_102024.csv), and prepare_tokenized_data.py, which applies RoBERTa tokenization and prepares the Hugging Face Dataset for training.

model.py:
Main training script using Hugging Face’s Trainer API and RoBERTa-bne model. Includes 5-fold cross-validation, logging, and evaluation.

prepare_tokenized_data.py:
Applies RoBERTa tokenizer and converts raw text into tokenized Hugging Face Dataset format.

es_hf_102024.csv:
Original Spanish hate speech/cyberbullying dataset sourced from Hugging Face

prepare_dataset.py:
Loads and filters the original dataset (es_hf_102024.csv) to clean or restructure the format.

augmented_dataset.csv:
Preprocessed and augmented version of the dataset. Includes class balancing through duplication of minority (cyberbullying) samples.







The training process was done on Google Colab. 

Steps to train the model on Google Colab:

Upload model.py, prepare_tokenized_data.py, augmented_dataset.csv, and requirement.txt files in the first cell

In the next cell, run prepare_tokenized_data.py 

After, in a different cell, run model.py
