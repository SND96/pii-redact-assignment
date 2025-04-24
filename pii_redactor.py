import openai
from typing import List, Dict, Union
import json
import os
import textwrap
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class PIIRedactor:
    def __init__(self, api_key: str = None, use_llama: bool = False):
        """
        Initialize the PII redactor.
        
        Args:
            api_key: OpenAI API key (required if use_llama is False)
            use_llama: Whether to use the local Llama model instead of OpenAI API
        """
        self.use_llama = use_llama
        if not use_llama:
            if not api_key:
                raise ValueError("OpenAI API key is required when not using Llama model")
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # Load the local Llama model
            model_path = "pii-redact-model-new"
            if not os.path.exists(model_path):
                raise ValueError(f"Llama model not found at {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float32
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pii_types = [
                    "TIME",
                    "USERNAME",
                    "IDCARD",
                    "SOCIALNUMBER",
                    "EMAIL",
                    "PASSPORT",
                    "DRIVERLICENSE",
                    "DOB",
                    "LASTNAME1",
                    "IP",
                    "GIVENNAME1",
                    "SEX",
                    "TEL",
                    "CITY",
                    "POSTCODE",
                    "STREET",
                    "STATE",
                    "BUILDING",
                    "TITLE",
                    "COUNTRY",
                    "DATE",
                    "PASS",
                    "SECADDRESS",
                    "LASTNAME2",
                    "GIVENNAME2",
                    "GEOCOORD",
                    "LASTNAME3"
        ]

    def recover_spans(self, text: str, pii_entities: list) -> list:
        """
        Given original text and LLM-extracted PII entities with only 'value' and 'label',
        find true character spans and return corrected entries.
        """
        corrected = []
        used_spans = set()
        
        for entity in pii_entities:
            value = entity["value"]
            label = entity["label"]

            start = text.find(value)
            if start == -1:
                # fallback: try a space-stripped search
                alt_value = value.replace(" ", "")
                for i in range(len(text)):
                    window = text[i:i+len(alt_value)]
                    if window.replace(" ", "") == alt_value:
                        start = i
                        break
            if start == -1:
                print(f"Warning: Couldn't find span for value: {value}")
                continue

            end = start + len(value)
            
            # Avoid duplicate matches (e.g. repeated names)
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
    
    def redact_text(self, text: str) -> Dict:
        """
        Redact PII from the given text using either OpenAI API or local Llama model.
        Returns the redacted text and the identified PII entities.
        """
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

        Respond in **valid JSON**. Do not include any other explanation.

        {textwrap.dedent(few_shot_example)}

        Now analyze this text:
        Input: \"\"\"{text}\"\"\"
        Output:
        """
            
        try:
            if self.use_llama:
                print("Using Llama model")
                # Prepare input for Llama model
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Generate response
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0,
                    do_sample=False
                )
                
                # Decode response
                content = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract JSON from response
                content = content.split("Output:")[-1].strip()
            else:
                # Use OpenAI API
                print("Using OpenAI API")
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a PII detection and redaction assistant. You must respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    response_format={ "type": "json_object" }
                )
                content = response.choices[0].message.content
            
            # Get the content and ensure it's a string
            if not isinstance(content, str):
                raise ValueError("Response content is not a string")
                
            # Try to parse the JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON response: {content}")
                raise e
            
            # Validate the result structure
            if not isinstance(result, dict):
                raise ValueError("Response is not a dictionary")
            if "redacted_text" not in result or "pii_entities" not in result:
                raise ValueError("Response missing required fields")
            result["pii_entities"] = self.recover_spans(text, result["pii_entities"])
            return result
            
        except Exception as e:
            print(f"Error during PII redaction: {str(e)}")
            return {
                "redacted_text": text,
                "pii_entities": []
            }

    def process_dataset(self, split: str = "train", batch_size: int = 10, limit: int = None) -> List[Dict]:
        """
        Process the PII dataset from local files and redact PII from the source_text column.
        
        Args:
            split: Dataset split to process (train, validation)
            batch_size: Number of examples to process at once
            limit: Maximum number of examples to process (None for all)
            
        Returns:
            List of dictionaries containing redacted text and PII entities
        """
        # Get the path to the split directory
        split_dir = Path("data") / split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
            
        # Load the dataset from disk
        dataset = load_from_disk(str(split_dir))
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
            
        results = []
        for example in dataset:
            # Process source text
            source_text = example["source_text"]
            result = self.redact_text(source_text)
            
            # Compare with existing privacy mask
            privacy_mask = example["privacy_mask"]
            span_labels = example["span_labels"]
            
            results.append({
                "id": example["id"],
                "language": example["language"],
                "set": example["set"],
                "original_source_text": source_text,
                "original_target_text": example["target_text"],
                "original_privacy_mask": privacy_mask,
                "original_span_labels": span_labels,
                "redacted_text": result["redacted_text"],
                "pii_entities": result["pii_entities"],
                "mbert_text_tokens": example["mbert_text_tokens"],
                "mbert_bio_labels": example["mbert_bio_labels"]
            })
        
        return results

def main():
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    use_llama = os.getenv("USE_LLAMA", "false").lower() == "true"
    
    if not use_llama and not api_key:
        print("Please set the OPENAI_API_KEY in your .env file")
        return
    
    redactor = PIIRedactor(api_key=api_key, use_llama=use_llama)
    
    # Process both train and validation splits
    for split in ["val"]:
        print(f"\nProcessing {split} split...")
        results = redactor.process_dataset(
            split=split,
            batch_size=10,
            limit=5  # Process only first 20 examples for demonstration
        )
        
        # Save results to a JSON file
        output_file = f"pii_redaction_results_{split}.json"
        print(f"Saving results to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        # Print results
        for i, result in enumerate(results):
            print(f"\nExample {i + 1} (ID: {result['id']}, Language: {result['language']}, Set: {result['set']}):")
            print("Original source text:", result["original_source_text"])
            print("Original target text:", result["original_target_text"])
            print("Redacted text:", result["redacted_text"])
            print("Original privacy mask:", result["original_privacy_mask"])
            print("Original span labels:", result["original_span_labels"])
            print("Identified PII entities:")
            for entity in result["pii_entities"]:
                print(entity)

if __name__ == "__main__":
    main() 