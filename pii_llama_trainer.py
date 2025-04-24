from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import logging
import os
from typing import List, Dict
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directories if they don't exist
os.makedirs("./pii-redact-model", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

try:
    # Load the dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("ai4privacy/pii-masking-300k")
    
    # Select 10% of the data
    logger.info("Selecting 10% of the data...")
    train_size = len(dataset["train"])
    val_size = len(dataset["validation"])
    train_subset = dataset["train"].select(range(int(train_size * 0.001)))
    val_subset = dataset["validation"].select(range(int(val_size * 0.001)))
    
    # Create new dataset dict with subsets
    dataset = {
        "train": train_subset,
        "validation": val_subset
    }
    
    logger.info(f"Dataset loaded successfully. Train size: {len(dataset['train'])}, Validation size: {len(dataset['validation'])}")

    # Initialize tokenizer and model
    logger.info("Initializing tokenizer and model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float32  # Changed to float32 for MPS compatibility
    )

    # Configure LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Get PEFT model
    model = get_peft_model(base_model, peft_config)
    logger.info("Model initialized successfully")

    def prepare_text(example):
        """Prepare text for training by combining source and target text."""
        # Create a prompt template
        prompt = f"""Below is a text containing Personally Identifiable Information (PII). 
Your task is to redact the PII by replacing it with appropriate tags.

Input text:
{example['source_text']}

Redacted text:
{example['target_text']}"""
        
        return {"text": prompt}

    # Prepare the dataset
    logger.info("Preparing dataset...")
    processed_dataset = {
        "train": dataset["train"].map(
            prepare_text,
            remove_columns=dataset["train"].column_names
        ),
        "validation": dataset["validation"].map(
            prepare_text,
            remove_columns=dataset["validation"].column_names
        )
    }

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

    tokenized_dataset = {
        "train": processed_dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        ),
        "validation": processed_dataset["validation"].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
    }

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./pii-redact-model",
        per_device_train_batch_size=4,  # Reduced batch size
        per_device_eval_batch_size=4,   # Reduced batch size
        eval_strategy="epoch",
        logging_dir="./logs",
        num_train_epochs=3,
        learning_rate=2e-4,
        save_total_limit=2,
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="tensorboard",
        gradient_accumulation_steps=4,  # Added gradient accumulation
        # Removed fp16=True as it's not supported on MPS
    )

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed successfully")

    # Save the model
    logger.info("Saving model...")
    trainer.save_model("./pii-redact-model-final")
    logger.info("Model saved successfully")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise