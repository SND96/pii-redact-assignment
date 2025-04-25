# PII Redaction Project

This repository contains tools and scripts for Personally Identifiable Information (PII) detection and redaction using various approaches, including fine-tuned Mistral models and OpenAI's API.

## Project Structure

- `pii_redactor.py`: Main script for PII detection and redaction
- `pii_mistral_fine_tune.py`: Script for fine-tuning Mistral model for PII detection
- `pii_eval.py`: Evaluation script for PII detection performance
- `dataset_create.py`: Script for downloading and preparing the PII masking dataset
- `default_pii_types.py`: Configuration file for PII types
- `tests/`: Directory containing test files
- `model_mistral/`: Directory containing fine-tuned Mistral model

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- CUDA-capable GPU (recommended for fine-tuning)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SND96/pii-redact-assignment
cd fathom-pii-project
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### 1. Dataset Preparation

To download and prepare the PII masking dataset:

```bash
python dataset_create.py
```

This script:
1. Downloads the "ai4privacy/pii-masking-300k" dataset from Hugging Face
2. Creates a `data` directory if it doesn't exist
3. Saves the train and validation splits to disk
4. Prints dataset statistics and a sample row

The dataset contains the following fields:
- `id`: Unique identifier
- `language`: Language of the text
- `set`: Dataset split (train/validation)
- `source_text`: Original text with PII
- `target_text`: Masked text with PII replaced
- `privacy_mask`: PII masking information
- `span_labels`: PII span labels
- `mbert_text_tokens`: Tokenized text for mBERT
- `mbert_bio_labels`: BIO labels for mBERT


### 2. Model Fine-tuning

To fine-tune the Mistral model using LoRA (Low-Rank Adaptation):

```bash
python pii_mistral_fine_tune.py
```

The script uses the following default configuration:
- Base model: "mistralai/Mistral-7B-Instruct-v0.2"
- Dataset: "ai4privacy/pii-masking-300k"
- Training size: 1000 examples
- Validation size: 100 examples
- LoRA configuration:
  - Rank (r): 8
  - Alpha: 32
  - Dropout: 0.05
  - Target modules: ["q_proj", "v_proj"]
- Training arguments:
  - Batch size: 4
  - Epochs: 3
  - Weight decay: 0.01
  - Save strategy: epoch
  - Evaluation steps: 500

### 3. PII Redaction

To run PII redaction on a text file or directory:

```bash
python pii_redactor.py --input input.txt --output output.json
```

Command line arguments:
- `--input` or `-i`: Input file or directory containing text to redact (required)
- `--output` or `-o`: Output file to save redacted results (required)
- `--use-mistral`: Use local Mistral model instead of OpenAI API
- `--model-path`: Path to the local Mistral model directory (default: "model_mistral")
- `--limit`: Maximum number of examples to process
- `--pii-types-file`: Path to file containing PII types (one per line)
- `--pii-types`: List of PII types to use for redaction

Example using Mistral model:
```bash
python pii_redactor.py --input input.txt --output output.json --use-mistral --model-path model_mistral
```

Example using OpenAI with custom PII types:
```bash
python pii_redactor.py --input input.txt --output output.json --pii-types NAME EMAIL PHONE
```

### 3. Evaluation

To evaluate PII detection performance:

```bash
python pii_eval.py --file pii_redaction_results_val.json
```

Command line arguments:
- `--file`: Path to the JSON file containing redaction results (default: 'pii_redaction_results_val.json')
- `--per-label`: Include per-label metrics in the results
- `--ignore-labels`: Only match spans based on start and end positions, ignoring labels

The evaluation script provides the following metrics:
- Overall metrics:
  - Precision
  - Recall
  - F1 score
- Per-label metrics (when `--per-label` is specified):
  - Precision, Recall, and F1 for each PII type
  - Counts of:
    - True Positives
    - False Positives
    - False Negatives

Example with per-label metrics:
```bash
python pii_eval.py --file output.json --per-label
```



## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Output Format

The output JSON file contains an array of objects with the following structure:
```json
[
  {
    "original_text": "original text content",
    "redacted_text": "text with PII redacted",
    "pii_entities": [
      {
        "label": "PII_TYPE",
        "value": "original PII value",
        "start": start_index,
        "end": end_index
      }
    ]
  }
]
```

