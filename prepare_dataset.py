# Uses Hugging Face Transformers pipeline to paraphrase 30% of positive samples
# Last updated: 2025-06-28

# Import necessary libraries
import pandas as pd  # for loading and manipulating dataframes
from transformers import pipeline  # for using pretrained NLP pipelines
import random  # for reproducible random sampling

# Load the original dataset
df = pd.read_csv("es_hf_102024.csv")
print("Original dataset size:", len(df))

# Select 30% of positive (cyberbullying) examples for augmentation
positive_df = df[df["labels"] == 1].copy()
sample_size = int(0.3 * len(positive_df))
paraphrase_subset = positive_df.sample(n=sample_size, random_state=42)

# Load a paraphrasing model from Hugging Face
print("Loading paraphrasing model...")
paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

# Generate paraphrased texts using the model
augmented_texts = []
for text in paraphrase_subset["text"]:
    prompt = f"paraphrase: {text} </s>"
    try:
        para = paraphraser(prompt, max_length=256, do_sample=True, top_k=120, top_p=0.98, num_return_sequences=1)
        augmented_texts.append(para[0]['generated_text'])
    except Exception as e:
        print(f"Paraphrasing failed for: {text}\nError: {e}")
        augmented_texts.append(text)  # Use original text if paraphrasing fails

# Package the paraphrased samples into a new DataFrame
paraphrased_df = pd.DataFrame({
    "text": augmented_texts,
    "labels": 1,  # maintain the cyberbullying label
    "source": "paraphrased",
    "dataset": "augmented",
    "nb_annotators": 1,
    "tweet_id": ["aug_" + str(i) for i in range(len(augmented_texts))],
    "post_author_country_location": "N/A"
})

# Combine the original and augmented data into a single dataset
df_augmented = pd.concat([df, paraphrased_df], ignore_index=True)
print("Augmented dataset size:", len(df_augmented))

# Save the final augmented dataset to a CSV file
df_augmented.to_csv("augmented_dataset.csv", index=False)
print("Saved to augmented_dataset.csv")