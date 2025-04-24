from datasets import load_dataset
import os

# Load the dataset
dataset = load_dataset("ai4privacy/pii-masking-300k")

# Create output directory if it doesn't exist
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Get the train and validation splits
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# Save the splits
train_dataset.save_to_disk(os.path.join(output_dir, "train"))
val_dataset.save_to_disk(os.path.join(output_dir, "val"))

print(f"Dataset splits saved to {output_dir}/")
print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print("\nSample row from train set:")
print(f"ID: {train_dataset[0]['id']}")
print(f"Language: {train_dataset[0]['language']}")
print(f"Set: {train_dataset[0]['set']}")
print("\nSource text (with PII):")
print(train_dataset[0]['source_text'])
print("\nTarget text (masked):")
print(train_dataset[0]['target_text'])






