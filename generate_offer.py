"""
Generate structured JSON offers from raw promotional text.
This script loads a fine-tuned GPT2 model and uses it to convert promotional text into structured JSON format.
"""

import os
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

class OfferGenerator:
    def __init__(self, model_path):
        """
        Initialize the offer generator with a fine-tuned model.
        
        Args:
            model_path: Path to the directory containing the fine-tuned model files
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        
        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_offer(self, input_text, temperature=0.7, max_length=512):
        """
        Generate a structured JSON offer from raw promotional text.
        
        Args:
            input_text: Raw promotional text
            temperature: Sampling temperature for generation
            max_length: Maximum length of generated text
            
        Returns:
            str: Generated JSON offer
        """
        # Format input with the expected prompt structure
        prompt = f"Input: {input_text}\nOutput:"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Generate output
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        # Decode and clean up the output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the JSON part
        if "Output:" in generated_text:
            generated_text = generated_text.split("Output:")[1].strip()
        
        # Try to extract valid JSON using regex
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                # Validate JSON
                json_obj = json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                return generated_text
        
        return generated_text

    def parse_json(self, text):
        """
        Parse the generated text to extract valid JSON.
        
        Args:
            text: Generated text that might contain JSON
            
        Returns:
            dict: Parsed JSON object or None if parsing fails
        """
        try:
            # Try to extract JSON using regex
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            return None
        except (json.JSONDecodeError, AttributeError):
            return None

def load_model(model_path):
    """
    Helper function to load the model and create an offer generator.
    
    Args:
        model_path: Path to the directory containing model files
        
    Returns:
        OfferGenerator: Initialized offer generator
    """
    return OfferGenerator(model_path)

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate structured JSON offers from promotional text")
    parser.add_argument("--model_path", type=str, default="model", help="Path to the fine-tuned model")
    parser.add_argument("--input", type=str, required=True, help="Input promotional text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature (0.1-1.0)")
    
    args = parser.parse_args()
    
    generator = load_model(args.model_path)
    result = generator.generate_offer(args.input, temperature=args.temperature)
    
    print(result)
