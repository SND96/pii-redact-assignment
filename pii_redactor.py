import json
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Union

import openai
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as transformers_logging
import argparse
from static.default_pii_types import DEFAULT_PII_TYPES

# Suppress tokenizer warnings
transformers_logging.set_verbosity_error()

class PIIRedactor:
    """A class for redacting Personally Identifiable Information (PII) from text using either OpenAI API or local Mistral model."""
    
    def __init__(
        self, 
        api_key: str = None, 
        use_mistral: bool = False, 
        model_path: str = "model_mistral",
        pii_types: List[str] = None,
        pii_types_file: str = None
    ):
        """
        Initialize the PII redactor.
        
        Args:
            api_key: OpenAI API key (required if use_mistral is False)
            use_mistral: Whether to use the local Mistral model instead of OpenAI API
            model_path: Path to the local Mistral model directory
            pii_types: List of PII types to use for redaction
            pii_types_file: Path to a file containing PII types (one per line)
        """
        self.use_mistral = use_mistral
        if not use_mistral:
            if not api_key:
                raise ValueError("OpenAI API key is required when not using Mistral model")
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self._setup_mistral_model(model_path)
            
        # Load PII types
        if pii_types is not None:
            self.pii_types = pii_types
        elif pii_types_file is not None:
            self.pii_types = self._load_pii_types_from_file(pii_types_file)
        else:
            self.pii_types = DEFAULT_PII_TYPES

    def _load_pii_types_from_file(self, file_path: str) -> List[str]:
        """Load PII types from a file (one type per line)."""
        try:
            with open(file_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error loading PII types from file: {str(e)}")
            return DEFAULT_PII_TYPES

    def _setup_mistral_model(self, model_path: str) -> None:
        """Set up the Mistral model for local inference."""
        if not os.path.exists(model_path):
            raise ValueError(f"Mistral model not found at {model_path}. Please download the model and place it in the {model_path} directory.")
        
        try:
            self.device = self._get_device()
            print(f"Using device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)
            print(f"Mistral model successfully loaded from {model_path}")
        except Exception as e:
            print(f"Error loading Mistral model: {str(e)}")
            raise
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_device(self) -> torch.device:
        """Determine the best available device for model inference."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def align_redaction(self, original: str, redacted: str) -> List[Dict]:
        """
        Align redacted text with original text to extract PII entities.
        
        Args:
            original: The original text containing PII
            redacted: The redacted text with PII replaced by [LABEL] tokens
            
        Returns:
            List of dictionaries containing PII entities with their spans
        """
        entities = []
        PLACEHOLDER_RE = re.compile(r"\[([A-Z]+)\]")

        i_r = i_o = 0  # indices in redacted / original
        while i_r < len(redacted):
            match = PLACEHOLDER_RE.match(redacted, i_r)
            if not match:
                i_r += 1
                i_o += 1
                continue

            label = match.group(1)
            i_r = match.end()

            next_r_chr = redacted[i_r:i_r+1]
            end_o = original.find(next_r_chr, i_o) if next_r_chr else len(original)
            if end_o == -1:
                end_o = len(original)

            value = original[i_o:end_o]
            entities.append({
                "label": label,
                "value": value,
                "start": i_o,
                "end": end_o
            })

            i_o = end_o

        return entities

    def recover_spans(self, text: str, pii_entities: list) -> list:
        """
        Recover character spans for PII entities in the original text.
        
        Args:
            text: The original text
            pii_entities: List of PII entities with 'value' and 'label'
            
        Returns:
            List of PII entities with correct character spans
        """
        corrected = []
        used_spans = set()

        for entity in pii_entities:
            value = entity["value"]
            label = entity["label"]

            start = text.find(value)
            if start == -1:
                start = self._find_approximate_span(text, value)
                if start == -1:
                    print(f"Warning: Couldn't find span for value: {value}")
                    continue

            end = start + len(value)
            if (start, end) in used_spans:
                continue
                
            used_spans.add((start, end))
            corrected.append({
                "label": label,
                "value": value,
                "start": start,
                "end": end
            })

        return corrected

    def _find_approximate_span(self, text: str, value: str) -> int:
        """Find approximate span for a value by trying space-stripped search."""
        alt_value = value.replace(" ", "")
        for i in range(len(text)):
            window = text[i:i+len(alt_value)]
            if window.replace(" ", "") == alt_value:
                return i
        return -1

    def redact_text(self, text: str) -> Dict:
        """
        Redact PII from the given text using either OpenAI API or local Mistral model.
        
        Args:
            text: The text to redact
            
        Returns:
            Dictionary containing redacted text and identified PII entities
        """
        if self.use_mistral:
            return self._redact_with_mistral(text)
        return self._redact_with_openai(text)

    def _redact_with_mistral(self, text: str) -> Dict:
        """Redact PII using the local Mistral model."""
        prompt = f"""<|system|>\nYou redact PII from input and return redacted version only. Only use the following PII types: {', '.join(self.pii_types)}. \n<|user|>\n{text}\n<|assistant|>\n"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True
            )
            
            content = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            redacted_text = self._extract_assistant_response(content)
            
            return {
                "redacted_text": redacted_text,
                "pii_entities": self.align_redaction(text, redacted_text)
            }
            
        except Exception as e:
            print(f"Error during PII redaction with Mistral: {str(e)}")
            return {
                "redacted_text": text,
                "pii_entities": []
            }

    def _extract_assistant_response(self, content: str) -> str:
        """Extract the assistant's response from the model output."""
        start_marker = "<|assistant|>"
        end_marker = "<|user|>"
        
        start_idx = content.find(start_marker)
        if start_idx != -1:
            start_idx += len(start_marker)
            end_idx = content.find(end_marker, start_idx)
            if end_idx == -1:
                end_idx = len(content)
            return content[start_idx:end_idx].strip()
        return content.strip()

    def _redact_with_openai(self, text: str) -> Dict:
        """Redact PII using the OpenAI API."""
        few_shot_example = """
        Example:
        Input: "John Smith's phone number is (555) 123-4567 and his email is john.smith@example.com."
        Output:
        {
        "redacted_text": "[NAME]'s phone number is [PHONE] and his email is [EMAIL].",
        "pii_entities": [
            {
            "label": "NAME",
            "value": "John Smith",
            "start": 0,
            "end": 10
            },
            {
            "label": "PHONE",
            "value": "(555) 123-4567",
            "start": 30,
            "end": 45
            },
            {
            "label": "EMAIL",
            "value": "john.smith@example.com",
            "start": 63,
            "end": 86
            }
        ]
        }
        """

        prompt = f"""
        You are a data redaction assistant.

        Your task is to identify all Personally Identifiable Information (PII) in the following text and redact them.

        Redact each PII by replacing it with [label] (in uppercase, in square brackets).
        Only use the following PII types: {', '.join(self.pii_types)}

        For each PII entity found, return:
        - "label": the PII type (from the list above)
        - "value": the original text span
        - "start": the character index where the PII starts in the original text
        - "end": the character index where the PII ends (exclusive)
        Example:
        {textwrap.dedent(few_shot_example)}
        Respond in **valid JSON**. Do not include any other explanation.
        Input:
         \"\"\"{text}\"\"\"
         Output:
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a PII detection and redaction assistant. You must respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content)
            result["pii_entities"] = self.recover_spans(text, result["pii_entities"])
            return result
            
        except Exception as e:
            print(f"Error during PII redaction with OpenAI: {str(e)}")
            return {
                "redacted_text": text,
                "pii_entities": []
            }

    def process_dataset(self, input_path: str, limit: int = None) -> List[Dict]:
        """
        Process the PII dataset from input files and redact PII from the source_text column.
        
        Args:
            input_path: Path to input file or directory
            limit: Maximum number of examples to process (None for all)
            
        Returns:
            List of dictionaries containing redacted text and PII entities
        """
        input_path = Path(input_path)
        if self.use_mistral:
            print("Using Mistral")
        else:
            print("Using OpenAI")
        
        if input_path.is_file():
            # Process single file
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            result = self.redact_text(text)
            return [{
                "original_text": text,
                "redacted_text": result["redacted_text"],
                "pii_entities": result["pii_entities"]
            }]
        elif input_path.is_dir():
            # Process dataset directory
            dataset = load_from_disk(str(input_path))
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
                
            results = []
            
            for example in tqdm(dataset, desc="Processing dataset"):
                result = self.redact_text(example["source_text"])
                results.append({
                    "id": example["id"],
                    "language": example["language"],
                    "set": example["set"],
                    "original_source_text": example["source_text"],
                    "original_target_text": example["target_text"],
                    "original_privacy_mask": example["privacy_mask"],
                    "original_span_labels": example["span_labels"],
                    "redacted_text": result["redacted_text"],
                    "pii_entities": result["pii_entities"],
                    "mbert_text_tokens": example["mbert_text_tokens"],
                    "mbert_bio_labels": example["mbert_bio_labels"]
                })
            
            return results
        else:
            raise ValueError(f"Input path {input_path} does not exist or is not a file/directory")

def main():
    """Main function to run the PII redactor."""
    parser = argparse.ArgumentParser(description="PII Redactor - Redact Personally Identifiable Information from text")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input file or directory containing text to redact")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output file to save redacted results")
    parser.add_argument("--use-mistral", action="store_true", help="Use local Mistral model instead of OpenAI API")
    parser.add_argument("--model-path", type=str, default="model_mistral", help="Path to the local Mistral model directory")
    parser.add_argument("--limit", type=int, help="Maximum number of examples to process")
    parser.add_argument("--pii-types-file", type=str, help="Path to file containing PII types (one per line)")
    parser.add_argument("--pii-types", type=str, nargs="+", help="List of PII types to use for redaction")
    
    args = parser.parse_args()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not args.use_mistral and not api_key:
        print("Error: OPENAI_API_KEY environment variable is required when not using Mistral model")
        return
    
    redactor = PIIRedactor(
        api_key=api_key,
        use_mistral=args.use_mistral,
        model_path=args.model_path,
        pii_types=args.pii_types,
        pii_types_file=args.pii_types_file
    )
    
    print(f"\nProcessing input: {args.input}")
    results = redactor.process_dataset(
        input_path=args.input,
        limit=args.limit
    )
    
    print(f"Saving results to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main() 