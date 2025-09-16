#!/usr/bin/env python3
"""Test script for IPC Section 375 specifically."""

from inference_engine import LegalQuestionAnswerer
import os
from project_structure import MODELS_DIR

def test_ipc_375():
    print("Testing IPC Section 375 with improved generation...")
    print("=" * 50)
    
    # Use the merged model
    model_path = os.path.join(MODELS_DIR, 'legal_qwen_merged')
    print(f"Loading model from: {model_path}")
    
    try:
        qa_system = LegalQuestionAnswerer(model_path)
        qa_system.load()
        print("✅ Model loaded successfully!")
        
        # Test different ways of asking about IPC 375
        questions = [
            "What is IPC Section 375?",
            "IPC Section 375",
            "Section 375 of Indian Penal Code",
            "What does Section 375 of IPC deal with?",
            "Explain IPC 375"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Test {i} ---")
            print(f"Question: {question}")
            
            try:
                result = qa_system.answer(question)
                answer = result["answer"]
                print(f"Answer: {answer}")
                
                # Check if the answer mentions rape
                if "rape" in answer.lower():
                    print("✅ Answer correctly mentions rape!")
                else:
                    print("❌ Answer doesn't mention rape")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    test_ipc_375()
