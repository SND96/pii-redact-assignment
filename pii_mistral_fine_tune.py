from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os
from typing import Optional, Dict, Any

class PIITrainer:
    def __init__(
        self,
        model_checkpoint: str = "mistralai/Mistral-7B-Instruct-v0.2",
        dataset_name: str = "ai4privacy/pii-masking-300k",
        output_dir: str = "./tinyllama-pii-redactor-lora",
        train_size: int = 1000,
        val_size: int = 100,
        lora_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None
    ):
        self.model_checkpoint = model_checkpoint
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.train_size = train_size
        self.val_size = val_size
        
        # Default LoRA config
        self.lora_config = lora_config or {
            "r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "v_proj"]
        }
        
        # Default training args
        self.training_args = training_args or {
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "num_train_epochs": 3,
            "weight_decay": 0.01,
            "save_total_limit": 2,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "eval_steps": 500,
            "logging_first_step": True
        }
        
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _format_chat_prompt(self, source: str, target: str) -> str:
        return f"<|system|>\nYou are a data redaction assistant. You redact PII in the input below and return only the redacted text. Do not explain anything.\n<|user|>\n{source}\n<|assistant|>\n{target}"

    def _format_prompt_prefix(self, source: str) -> str:
        return f"<|system|>\nYou are a data redaction assistant. You redact PII in the input below and return only the redacted text. Do not explain anything.\n<|user|>\n{source}\n<|assistant|>\n"

    def _tokenize_function(self, batch):
        all_data = []
        for source, target in zip(batch["source_text"], batch["target_text"]):
            full_prompt = self._format_chat_prompt(source, target)
            prompt_prefix = self._format_prompt_prefix(source)

            tokenized = self.tokenizer(full_prompt, truncation=True, max_length=512, padding="max_length")
            prefix_len = len(self.tokenizer(prompt_prefix, truncation=True, max_length=512)["input_ids"])

            # Mask everything except assistant response
            labels = tokenized["input_ids"].copy()
            labels[:prefix_len] = [-100] * prefix_len
            labels = [
                -100 if token_id == self.tokenizer.pad_token_id else token_id
                for token_id in labels
            ]

            tokenized["labels"] = labels
            all_data.append(tokenized)

        # Convert list of dicts to dict of lists
        keys = all_data[0].keys()
        return {k: [d[k] for d in all_data] for k in keys}

    def setup(self):
        """Initialize the model, tokenizer, and load the dataset"""
        # Load dataset
        dataset = load_dataset(self.dataset_name)
        train_dataset = dataset["train"].select(range(self.train_size))
        val_dataset = dataset["validation"].select(range(self.val_size))

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, trust_remote_code=True)
        
        # Fix missing pad_token error
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_checkpoint,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )
        
        # Prepare model for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        lora_config = LoraConfig(**self.lora_config)
        self.model = get_peft_model(self.model, lora_config)

        # Tokenize datasets
        tokenized_train = train_dataset.map(self._tokenize_function, batched=True, remove_columns=train_dataset.column_names)
        tokenized_val = val_dataset.map(self._tokenize_function, batched=True, remove_columns=val_dataset.column_names)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            fp16=torch.cuda.is_available(),
            logging_dir=os.path.join(self.output_dir, "logs"),
            **self.training_args
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

    def train(self):
        """Start the training process"""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")
        
        self.trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

def main():
    # Example usage
    trainer = PIITrainer(
        train_size=1000,
        val_size=100,
        output_dir="./tinyllama-pii-redactor-lora"
    )
    trainer.setup()
    trainer.train()

if __name__ == "__main__":
    main()