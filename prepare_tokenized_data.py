from datasets import Dataset  # Import Hugging Face's Dataset class to convert pandas DataFrame to Dataset
import pandas as pd  # Import pandas for handling CSV file
from transformers import AutoTokenizer  # Import tokenizer utility to preprocess text for model input

# Load the CSV dataset into a pandas DataFrame
df = pd.read_csv("augmented_dataset.csv")  # Read the augmented dataset from a CSV file

# Initialize the tokenizer using a pretrained Spanish RoBERTa model
model_name = "PlanTL-GOB-ES/roberta-base-bne"  # Specify the name of the pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load the tokenizer from Hugging Face

# Define the tokenization function to preprocess each example
def tokenize_function(examples):
    # Tokenize the text column, applying truncation and padding to a fixed length
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Convert the pandas DataFrame to a Hugging Face Dataset format
raw_dataset = Dataset.from_pandas(df)  # Create a raw dataset from the DataFrame

# Apply the tokenizer function to all entries in the dataset
tokenized = raw_dataset.map(tokenize_function, batched=True)  # Tokenize all texts in batches

# Save the processed tokenized dataset to disk for later use in training
tokenized.save_to_disk("tokenized_dataset")  # Save the tokenized dataset to a local folder