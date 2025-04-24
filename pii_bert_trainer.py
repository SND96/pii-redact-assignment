from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate
from typing import List, Dict
import json
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directories if they don't exist
os.makedirs("./mbert-pii", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

try:
    # Load the dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("ai4privacy/pii-masking-300k")
    logger.info(f"Dataset loaded successfully. Train size: {len(dataset['train'])}, Validation size: {len(dataset['validation'])}")

    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

    def extract_unique_labels(dataset_split):
        """Extract unique labels from the dataset."""
        labels = {'O'}  # Start with 'O' for Outside
        for row in dataset_split:
            if 'privacy_mask' in row:
                for mask in row['privacy_mask']:
                    # Add both B- and I- prefixed versions of each label
                    label = mask['label']
                    labels.add(f"B-{label}")
                    labels.add(f"I-{label}")
        return sorted(list(labels))

    # Get unique labels and create label maps
    logger.info("Extracting unique labels...")
    label_list = extract_unique_labels(dataset["train"])
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    logger.info(f"Found {len(label_list)} unique labels: {label_list}")

    def tokenize_and_align_labels(example):
        """Tokenize text and align labels with tokens."""
        # Tokenize the text
        tokenized = tokenizer(
            example["source_text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            is_split_into_words=False,
        )
        
        # Initialize all labels as 'O'
        labels = ['O'] * len(tokenized['input_ids'])
        
        # Align labels with tokens
        if 'privacy_mask' in example:
            word_ids = tokenized.word_ids()
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    continue
                    
                # Check if this word is part of a PII span
                for mask in example['privacy_mask']:
                    if word_idx >= mask['start'] and word_idx < mask['end']:
                        if word_idx != previous_word_idx:
                            # Start of a new entity
                            labels[word_idx] = f"B-{mask['label']}"
                        else:
                            # Inside of an entity
                            labels[word_idx] = f"I-{mask['label']}"
                        break
                
                previous_word_idx = word_idx
        
        # Convert labels to IDs
        tokenized["labels"] = [label_to_id[label] for label in labels]
        
        return tokenized

    # Tokenize the dataset
    logger.info("Tokenizing dataset...")
    tokenized_ds = dataset.map(
        tokenize_and_align_labels,
        batched=False,
        remove_columns=dataset["train"].column_names
    )
    logger.info("Dataset tokenization completed")

    # Load the model
    logger.info("Loading model...")
    model = BertForTokenClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    logger.info("Model loaded successfully")

    # Load evaluation metric
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        """Compute metrics for evaluation."""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./mbert-pii",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        report_to="tensorboard",
    )

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed successfully")

    # Save the model
    logger.info("Saving model...")
    trainer.save_model("./mbert-pii-final")
    logger.info("Model saved successfully")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise