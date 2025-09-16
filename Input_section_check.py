#!/usr/bin/env python3
"""Test script for IPC Section 375 specifically."""
input_ipc = input("Enter the IPC section you want to test: ")

from inference_engine import LegalQuestionAnswerer
import os
from project_structure import MODELS_DIR
from ui_utils import toast, loading
def test_ipc_375():
    print(f"Testing IPC Section {input_ipc} with improved generation...")
    print("=" * 50)
    
    # Use the merged model
    model_path = os.path.join(MODELS_DIR, 'legal_qwen_merged')
    print(f"Loading model from: {model_path}")
    
    try:
        with loading("Loading model..."):
            qa_system = LegalQuestionAnswerer(model_path)
            qa_system.load()
        toast("Model loaded successfully", type="success")
        
        # Test different ways of asking about IPC 375
        questions = [
            f"What is IPC Section {input_ipc}?",
            f"IPC Section {input_ipc}",
            f"Section {input_ipc} of Indian Penal Code",
            f"What does Section {input_ipc} of IPC deal with?",
            f"Explain IPC {input_ipc}"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Test {i} ---")
            print(f"Question: {question}")
            
            try:
                with loading("Thinking..."):
                    result = qa_system.answer(question)
                answer = result["answer"]
                print(f"Answer: {answer}")
                
                # Check if the answer mentions rape
                if "rape" in answer.lower():
                    print("✅ Answer correctly mentions rape!")
                else:
                    print("❌ Answer doesn't mention rape")
                    
            except Exception as e:
                toast(f"Error: {e}", type="error")
                
    except Exception as e:
        toast(f"Error loading model: {e}", type="error")

if __name__ == "__main__":
    test_ipc_375()
    