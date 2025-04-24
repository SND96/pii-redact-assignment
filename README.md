# PII Redaction Project

A Python-based project for detecting and redacting Personally Identifiable Information (PII) from text using both OpenAI's API and local Llama models.

## Features

- PII detection and redaction using OpenAI's API
- Local PII redaction using fine-tuned Llama models
- Support for processing datasets in batches
- Evaluation metrics for PII redaction performance
- Training scripts for fine-tuning models

## Project Structure

- `pii_redactor.py`: Main class for PII detection and redaction
- `pii_eval.py`: Evaluation scripts for PII redaction performance
- `pii_bert_trainer.py`: Training script for BERT-based models
- `pii_llama_trainer.py`: Training script for Llama models
- `dataset_create.py`: Script for creating training datasets
- `generate_data.ipynb`: Jupyter notebook for data generation
- `pii-redact-model/`: Directory containing trained models
- `data/`: Directory for storing datasets
- `logs/`: Directory for storing training logs

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`:
  - openai>=1.0.0
  - python-dotenv>=0.19.0
  - datasets>=2.0.0
  - tqdm>=4.65.0
  - torch>=2.0.0
  - transformers>=4.30.0
  - scikit-learn>=1.0.0
  - numpy>=1.21.0

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pii-redact-assignment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Using OpenAI API

```python
from pii_redactor import PIIRedactor

# Initialize with OpenAI API
redactor = PIIRedactor(api_key="your_api_key")

# Redact text
result = redactor.redact_text("Sample text containing PII like John Doe's email: john@example.com")
```

### Using Local Llama Model

```python
from pii_redactor import PIIRedactor

# Initialize with local Llama model
redactor = PIIRedactor(use_llama=True)

# Redact text
result = redactor.redact_text("Sample text containing PII like John Doe's email: john@example.com")
```

### Processing Datasets

```python
# Process a dataset in batches
results = redactor.process_dataset(split="train", batch_size=10, limit=100)
```

## Command Line Usage

### Evaluation Script
```bash
# Basic evaluation
python pii_eval.py --input-file pii_redaction_results.json

# Evaluate with per-label metrics
python pii_eval.py --input-file pii_redaction_results.json --per-label

# Evaluate ignoring label types (only position matching)
python pii_eval.py --input-file pii_redaction_results.json --ignore-labels
```

### Training Scripts

#### BERT Model Training
```bash
# Basic training
python pii_bert_trainer.py

# Training with custom parameters
python pii_bert_trainer.py --output-dir ./mbert-pii --num-epochs 5 --batch-size 16
```

#### Llama Model Training
```bash
# Basic training
python pii_llama_trainer.py

# Training with custom parameters
python pii_llama_trainer.py --output-dir ./pii-redact-model --num-epochs 3 --batch-size 8
```

### Dataset Creation
```bash
# Create training dataset
python dataset_create.py --output-file data/train_dataset.json --num-samples 1000
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

