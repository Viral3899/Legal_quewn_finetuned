# inference.py
"""Inference engine for the fine-tuned legal LLM."""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import logging
from typing import List, Dict, Optional
from config import MODEL_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalLMInference:
    """Inference engine for Legal Language Model."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_path: str = None):
        """Load the fine-tuned model and tokenizer."""
        if model_path:
            self.model_path = model_path
        
        if not self.model_path:
            # Look for the most recent merged model
            merged_models = [d for d in os.listdir(MODELS_DIR) if d.startswith("legal_qwen_merged")]
            if merged_models:
                self.model_path = os.path.join(MODELS_DIR, merged_models[-1])
            else:
                raise ValueError("No model path provided and no merged model found")
        
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Setup generation config
            self.generation_config = GenerationConfig(
                max_length=MODEL_CONFIG.max_length,
                temperature=MODEL_CONFIG.temperature,
                top_p=MODEL_CONFIG.top_p,
                do_sample=MODEL_CONFIG.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def format_prompt(self, question: str) -> str:
        """Format the question into the proper prompt format."""
        prompt = f"<|im_start|>system\nYou are a helpful assistant specializing in Indian legal matters. Provide accurate and informative answers to legal questions.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    
    def generate_response(self, question: str, max_new_tokens: int = 256) -> str:
        """Generate a response to a legal question."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Format the prompt
        prompt = self.format_prompt(question)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.generation_config.max_length - max_new_tokens
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode the response
            generated_text = self.tokenizer.decode(
                outputs.sequences[0], 
                skip_special_tokens=True
            )
            
            # Extract only the assistant's response
            response = generated_text.split("<|im_start|>assistant\n")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def batch_generate(self, questions: List[str], max_new_tokens: int = 256) -> List[str]:
        """Generate responses for multiple questions."""
        responses = []
        
        for question in questions:
            response = self.generate_response(question, max_new_tokens)
            responses.append(response)
        
        return responses
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        print("Legal AI Assistant - Interactive Chat")
        print("Type 'quit' or 'exit' to end the session.")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nYour legal question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Thank you for using Legal AI Assistant!")
                    break
                
                if not question:
                    continue
                
                print("\nGenerating response...")
                response = self.generate_response(question)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

class LegalQuestionAnswerer:
    """High-level interface for legal question answering."""
    
    def __init__(self, model_path: str = None):
        self.inference_engine = LegalLMInference(model_path)
        self.is_loaded = False
    
    def load(self):
        """Load the model."""
        self.inference_engine.load_model()
        self.is_loaded = True
    
    def answer(self, question: str) -> Dict[str, str]:
        """Answer a legal question and return structured response."""
        if not self.is_loaded:
            self.load()
        
        response = self.inference_engine.generate_response(question)
        
        return {
            "question": question,
            "answer": response,
            "model_path": self.inference_engine.model_path
        }
    
    def answer_multiple(self, questions: List[str]) -> List[Dict[str, str]]:
        """Answer multiple legal questions."""
        if not self.is_loaded:
            self.load()
        
        results = []
        for question in questions:
            result = self.answer(question)
            results.append(result)
        
        return results

def main():
    """Main function for standalone inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Legal LLM Inference")
    parser.add_argument("--model_path", type=str, help="Path to the fine-tuned model")
    parser.add_argument("--question", type=str, help="Legal question to answer")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference_engine = LegalLMInference(args.model_path)
    inference_engine.load_model()
    
    if args.interactive:
        # Start interactive chat
        inference_engine.interactive_chat()
    elif args.question:
        # Answer single question
        response = inference_engine.generate_response(args.question)
        print(f"Question: {args.question}")
        print(f"Answer: {response}")
    else:
        # Default: ask for question
        question = input("Enter your legal question: ")
        response = inference_engine.generate_response(question)
        print(f"Answer: {response}")

if __name__ == "__main__":
    main()