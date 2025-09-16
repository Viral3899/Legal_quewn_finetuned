#!/usr/bin/env python3
"""Simple command-line test for the legal Q&A system."""

import os
import sys
from project_structure import MODELS_DIR
from inference_engine import LegalQuestionAnswerer
from ui_utils import toast, loading

def main():
    print("Legal AI Assistant - Command Line Test")
    print("=" * 50)
    
    # Find available models
    models = [d for d in os.listdir(MODELS_DIR) if d.startswith('legal_qwen')]
    
    if not models:
        print("❌ No models found in the models directory.")
        print(f"Models directory: {MODELS_DIR}")
        return
    
    print(f"Found {len(models)} model(s):")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    # Use the latest model
    latest_model = sorted(models)[-1]
    model_path = os.path.join(MODELS_DIR, latest_model)
    print(f"\nUsing model: {latest_model}")
    
    try:
        # Initialize the Q&A system
        with loading("Loading model... (this may take a few minutes)"):
            qa_system = LegalQuestionAnswerer(model_path)
            qa_system.load()
        toast("Model loaded successfully", type="success")
        
        # Interactive Q&A loop
        print("\n" + "=" * 50)
        print("Ask your legal questions (type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            try:
                with loading("Thinking..."):
                    result = qa_system.answer(question)
                print(f"\n⚖️  Answer: {result['answer']}")
            except Exception as e:
                toast(f"Error: {str(e)}", type="error")
    
    except Exception as e:
        toast(f"Error loading model: {str(e)}", type="error")
        return

if __name__ == "__main__":
    main()
