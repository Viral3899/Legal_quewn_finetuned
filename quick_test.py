#!/usr/bin/env python3
"""Quick test for model loading."""

import os
import sys
from project_structure import MODELS_DIR

def test_model_loading():
    print("Testing model loading...")
    
    # Check if models exist
    models = [d for d in os.listdir(MODELS_DIR) if d.startswith('legal_qwen')]
    print(f"Available models: {models}")
    
    if not models:
        print("❌ No models found!")
        return
    
    # Try to import and load
    try:
        from inference_engine import LegalQuestionAnswerer
        
        model_path = os.path.join(MODELS_DIR, models[0])
        print(f"Loading model: {model_path}")
        
        qa_system = LegalQuestionAnswerer(model_path)
        qa_system.load()
        print("✅ Model loaded successfully!")
        
        # Test question
        question = "What is IPC Section 302?"
        result = qa_system.answer(question)
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()
