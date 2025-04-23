import openai
from typing import List, Dict, Union
import json
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import os

class PIIRedactor:
    def __init__(self, api_key: str):
        """Initialize the PII redactor with OpenAI API key."""
        self.client = openai.OpenAI(api_key=api_key)
        self.pii_types = [
            "name", "email", "phone number", "address", 
            "social security number", "credit card number",
            "date of birth", "passport number", "driver's license number", "password", "date", "ip_address"
        ]
    
    def redact_text(self, text: str) -> Dict:
        """
        Redact PII from the given text using OpenAI API.
        Returns the redacted text and the identified PII entities.
        """
        prompt = f"""
        Analyze the following text and identify all Personally Identifiable Information (PII).
        For each PII entity found, provide:
        1. The type of PII (from this list: {', '.join(self.pii_types)})
        2. The exact text that contains the PII
        3. The start and end position in the original text
        
        Text to analyze:
        {text}
        
        Respond in JSON format with the following structure:
        {{
            "redacted_text": "text with PII replaced with [REDACTED]",
            "pii_entities": [
                {{
                    "type": "type of PII",
                    "value": "original text",
                    "start": start_position,
                    "end": end_position
                }}
            ]
        }}
        
        IMPORTANT: Your response must be valid JSON. Do not include any additional text or explanations.
        """
        
        print(text)
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
            
            # Get the content and ensure it's a string
            content = response.choices[0].message.content
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
            
            return result
            
        except Exception as e:
            print(f"Error during PII redaction: {str(e)}")
            return {
                "redacted_text": text,
                "pii_entities": []
            }

    def process_dataset(self, split: str = "train", batch_size: int = 10, limit: int = None) -> List[Dict]:
        """
        Process the PII dataset and redact PII from the source_text column.
        
        Args:
            split: Dataset split to process (train, validation, test)
            batch_size: Number of examples to process at once
            limit: Maximum number of examples to process (None for all)
            
        Returns:
            List of dictionaries containing redacted text and PII entities
        """
        # Load the dataset
        dataset = load_dataset("ai4privacy/pii-masking-300k", split=split)
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        results = []
        # Process each example in the dataset
        for i in tqdm(range(len(dataset))):
            example = dataset[i]
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
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set the OPENAI_API_KEY in your .env file")
        return
    
    redactor = PIIRedactor(api_key)
    
    # Process the PII dataset
    print("Processing dataset: ai4privacy/pii-masking-300k")
    results = redactor.process_dataset(
        split="train",
        batch_size=5,
        limit=5  # Process only first example for demonstration
    )
    
    # Print results
    for i, result in enumerate(results):
        print(f"\nExample {i + 1} (ID: {result['id']}, Language: {result['language']}, Set: {result['set']}):")
        print("Original source text:", result["original_source_text"])
        print("Redacted text:", result["redacted_text"])
        print("Original privacy mask:", result["original_privacy_mask"])
        print("Original span labels:", result["original_span_labels"])
        print("Identified PII entities:")
        for entity in result["pii_entities"]:
            print(f"- {entity['type']}: {entity['value']}")

if __name__ == "__main__":
    main() 