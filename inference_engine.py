# inference.py
"""Inference engine for the fine-tuned legal LLM."""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
from safetensors.torch import load_file as load_safetensors
import logging
from typing import List, Dict, Optional
from project_structure import MODEL_CONFIG, MODELS_DIR
from legal_knowledge_base import legal_kb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalLMInference:
    """Inference engine for Legal Language Model."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        # Force CPU usage to avoid device mapping issues
        self.device = torch.device("cpu")
        
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
                # Import required modules
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
            
            # Check if this is a LoRA adapter or merged model
            is_lora_adapter = os.path.exists(os.path.join(self.model_path, "adapter_config.json"))
            
            if is_lora_adapter:
                logger.info("Detected LoRA adapter, loading base model and adapter...")
                # For LoRA adapters, we need to load the base model and then the adapter
                
                # Load base model first
                base_model_name = "Qwen/Qwen2-0.5B"  # Adjust this to your base model
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                    device_map=None,
                    attn_implementation="eager"
                )
                
                # Load LoRA adapter
                self.model = PeftModel.from_pretrained(self.model, self.model_path)
                # Ensure model is on CPU
                self.model.to(self.device)
                
                # Load tokenizer from the adapter directory
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    padding_side="left"
                )
            else:
                # Regular merged model
                logger.info("Loading merged model...")
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    padding_side="left"
                )
                
                # Try standard load first
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        trust_remote_code=True,
                        low_cpu_mem_usage=False,
                        device_map=None,
                        attn_implementation="eager"
                    )
                    self.model.to(self.device)
                except Exception as load_err:
                    logger.warning(f"Standard load failed ({load_err}). Falling back to safetensors manual load...")
                    # Manual load via config + safetensors to avoid meta tensor path
                    config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
                    self.model = AutoModelForCausalLM.from_config(
                        config,
                        torch_dtype=torch.float32,
                        attn_implementation="eager"
                    )
                    weights_path = os.path.join(self.model_path, "model.safetensors")
                    if not os.path.exists(weights_path):
                        raise FileNotFoundError(f"Weights file not found: {weights_path}")
                    # Load weights on CPU first to avoid meta tensor copy issues
                    state_dict = load_safetensors(weights_path, device="cpu")
                    missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                    if missing:
                        logger.warning(f"Missing weights when loading: {missing[:10]}{'...' if len(missing) > 10 else ''}")
                    if unexpected:
                        logger.warning(f"Unexpected weights when loading: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
                    # Move to target device after weights are materialized
                    self.model.to(self.device)
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Ensure model embeddings match tokenizer size
            try:
                vocab_size = len(self.tokenizer)
                if self.model.get_output_embeddings() is not None:
                    current_vocab = self.model.get_output_embeddings().weight.size(0)
                    if current_vocab != vocab_size:
                        self.model.resize_token_embeddings(vocab_size)
            except Exception as resize_err:
                logger.warning(f"Could not verify/resize token embeddings: {resize_err}")
            
            # Setup generation config with more conservative settings
            self.generation_config = GenerationConfig(
                max_length=MODEL_CONFIG.max_length,
                temperature=0.7,  # Lower temperature for more focused responses
                top_p=0.9,        # More focused sampling
                top_k=50,         # Limit vocabulary choices
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Higher penalty to avoid repetition
                no_repeat_ngram_size=3,
                early_stopping=True,     # Stop when EOS token is generated
                use_cache=True
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
            if "<|im_start|>assistant\n" in generated_text:
                response = generated_text.split("<|im_start|>assistant\n")[-1]
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0]
            else:
                # Fallback: extract everything after the user's question
                if "<|im_start|>user\n" in generated_text:
                    user_part = generated_text.split("<|im_start|>user\n")[1]
                    if "<|im_end|>" in user_part:
                        response = user_part.split("<|im_end|>")[1]
                    else:
                        response = user_part
                else:
                    response = generated_text
            
            # Clean up the response
            response = response.strip()
            
            # For specific IPC sections, always use knowledge base since model is unreliable
            question_lower = question.lower()
            if any(section in question_lower for section in ['302', '375', '304', '307', '420']):
                logger.info(f"Using knowledge base for IPC section question: {question}")
                kb_answer = legal_kb.format_answer(question)
                return kb_answer
            
            # Check if response is valid - if it doesn't contain relevant legal keywords, use knowledge base
            legal_keywords = ['murder', 'rape', 'section', 'punishment', 'imprisonment', 'fine', 'offence', 'penal', 'code', 'ipc', 'sexual', 'assault']
            has_legal_content = any(keyword in response.lower() for keyword in legal_keywords)
            
            if (len(response) < 20 or 
                response.lower() in ['system', 'user', 'assistant'] or 
                not has_legal_content):
                # Use knowledge base as fallback
                logger.info("Using knowledge base fallback for poor model response")
                kb_answer = legal_kb.format_answer(question)
                if "No matching legal information found" not in kb_answer:
                    return kb_answer
                # If knowledge base doesn't have it, try fallback generation
                return self.generate_response_fallback(question, max_new_tokens)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_response_fallback(self, question: str, max_new_tokens: int = 256) -> str:
        """Fallback generation method with different parameters."""
        try:
            prompt = self.format_prompt(question)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512  # Shorter context
            ).to(self.model.device)
            
            # Use more conservative generation settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.3,  # Very low temperature
                    top_p=0.8,
                    top_k=20,
                    do_sample=True,
                    repetition_penalty=1.3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response
            if "<|im_start|>assistant\n" in generated_text:
                response = generated_text.split("<|im_start|>assistant\n")[-1]
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0]
            else:
                response = generated_text[len(prompt):]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in fallback generation: {e}")
            return "I apologize, but I'm having difficulty generating a proper response. Please try rephrasing your question."
    
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