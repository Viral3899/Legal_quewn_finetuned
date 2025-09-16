#!/usr/bin/env python3
"""Quick test for model loading."""

import os
import sys
from project_structure import MODELS_DIR
from ui_utils import toast, loading

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
        
        with loading("Loading model..."):
            qa_system = LegalQuestionAnswerer(model_path)
            qa_system.load()
        toast("Model loaded successfully", type="success")
        
        # Test question
        question = "What is IPC Section 302?"
        result = qa_system.answer(question)
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        
    except Exception as e:
        toast(f"Error: {e}", type="error")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()
