# evaluator.py
"""Evaluation module for the fine-tuned legal LLM."""

import json
import os
import pandas as pd
from typing import List, Dict, Tuple
import logging
from datetime import datetime
from inference_engine import LegalQuestionAnswerer
from data_loader import DataLoader
from project_structure import DATA_DIR, LOGS_DIR
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalModelEvaluator:
    """Evaluator for Legal Language Model performance."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.qa_system = LegalQuestionAnswerer(model_path)
        self.evaluation_results = []
    
    def load_test_data(self, test_file: str = "test_data.jsonl") -> List[Dict]:
        """Load test data from JSONL file."""
        test_path = os.path.join(DATA_DIR, test_file)
        
        if not os.path.exists(test_path):
            logger.warning(f"Test file {test_path} not found. Creating sample test data...")
            return self._create_sample_test_data()
        
        test_data = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                test_data.append(data)
        
        logger.info(f"Loaded {len(test_data)} test samples")
        return test_data
    
    def _create_sample_test_data(self) -> List[Dict]:
        """Create sample test data for evaluation."""
        sample_questions = [
            {
                "prompt": "<|im_start|>system\nYou are a helpful assistant specializing in Indian legal matters. Provide accurate and informative answers to legal questions.<|im_end|>\n<|im_start|>user\nWhat is Section 302 of IPC?<|im_end|>\n<|im_start|>assistant\n",
                "completion": "Section 302 of the Indian Penal Code deals with punishment for murder. It states that whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.<|im_end|>",
                "text": "Sample legal question about IPC Section 302"
            },
            {
                "prompt": "<|im_start|>system\nYou are a helpful assistant specializing in Indian legal matters. Provide accurate and informative answers to legal questions.<|im_end|>\n<|im_start|>user\nWhat are the rights of an accused person?<|im_end|>\n<|im_start|>assistant\n",
                "completion": "An accused person has several fundamental rights including: 1) Right to remain silent, 2) Right to legal representation, 3) Right to be informed of charges, 4) Right to bail (in certain cases), 5) Right to fair trial, 6) Right against self-incrimination, 7) Right to cross-examine witnesses.<|im_end|>",
                "text": "Sample question about rights of accused"
            },
            {
                "prompt": "<|im_start|>system\nYou are a helpful assistant specializing in Indian legal matters. Provide accurate and informative answers to legal questions.<|im_end|>\n<|im_start|>user\nWhat is the difference between bail and anticipatory bail?<|im_end|>\n<|im_start|>assistant\n",
                "completion": "Bail is granted after arrest and detention, allowing temporary release pending trial. Anticipatory bail (Section 438 CrPC) is granted before arrest, providing protection from arrest in anticipation of being accused of a non-bailable offense.<|im_end|>",
                "text": "Sample question about bail types"
            }
        ]
        
        return sample_questions
    
    def extract_question_answer(self, test_item: Dict) -> Tuple[str, str]:
        """Extract question and expected answer from test item."""
        prompt = test_item.get('prompt', '')
        completion = test_item.get('completion', '')
        
        # Extract question from prompt
        if '<|im_start|>user\n' in prompt and '<|im_end|>' in prompt:
            question = prompt.split('<|im_start|>user\n')[1].split('<|im_end|>')[0].strip()
        else:
            question = prompt
        
        # Extract expected answer from completion
        if '<|im_end|>' in completion:
            expected_answer = completion.split('<|im_end|>')[0].strip()
        else:
            expected_answer = completion
        
        return question, expected_answer
    
    def evaluate_sample(self, question: str, expected_answer: str) -> Dict:
        """Evaluate a single sample."""
        logger.info(f"Evaluating question: {question[:50]}...")
        
        try:
            # Generate response
            result = self.qa_system.answer(question)
            generated_answer = result['answer']
            
            # Simple evaluation metrics
            evaluation = {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "answer_length": len(generated_answer.split()),
                "contains_legal_terms": self._contains_legal_terms(generated_answer),
                "is_relevant": self._is_relevant_response(question, generated_answer),
                "timestamp": datetime.now().isoformat()
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating sample: {e}")
            return {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": f"Error: {str(e)}",
                "answer_length": 0,
                "contains_legal_terms": False,
                "is_relevant": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def _contains_legal_terms(self, text: str) -> bool:
        """Check if the response contains legal terms."""
        legal_terms = [
            'section', 'ipc', 'crpc', 'constitution', 'court', 'judge', 'law', 'legal',
            'bail', 'arrest', 'punishment', 'fine', 'imprisonment', 'offense', 'accused',
            'defendant', 'plaintiff', 'evidence', 'witness', 'trial', 'rights', 'act',
            'code', 'provision', 'article', 'amendment', 'supreme court', 'high court'
        ]
        
        text_lower = text.lower()
        return any(term in text_lower for term in legal_terms)
    
    def _is_relevant_response(self, question: str, answer: str) -> bool:
        """Simple relevance check based on common words."""
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall'}
        
        question_words -= stop_words
        answer_words -= stop_words
        
        # Check if there's overlap
        overlap = question_words.intersection(answer_words)
        return len(overlap) > 0 or len(answer.strip()) > 10
    
    def evaluate_model(self, num_samples: int = 10) -> Dict:
        """Evaluate the model on test data."""
        logger.info(f"Starting model evaluation with {num_samples} samples...")
        
        # Load test data
        test_data = self.load_test_data()
        
        # Sample random test cases if we have more than num_samples
        if len(test_data) > num_samples:
            test_data = random.sample(test_data, num_samples)
        
        # Evaluate each sample
        results = []
        for i, test_item in enumerate(test_data):
            logger.info(f"Evaluating sample {i+1}/{len(test_data)}")
            question, expected_answer = self.extract_question_answer(test_item)
            evaluation = self.evaluate_sample(question, expected_answer)
            results.append(evaluation)
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(results)
        
        # Save results
        self._save_evaluation_results(results, summary)
        
        return {
            "summary": summary,
            "detailed_results": results
        }
    
    def _calculate_summary_stats(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics from evaluation results."""
        total_samples = len(results)
        
        if total_samples == 0:
            return {"total_samples": 0}
        
        # Calculate metrics
        avg_answer_length = sum(r['answer_length'] for r in results) / total_samples
        legal_terms_percentage = sum(1 for r in results if r['contains_legal_terms']) / total_samples * 100
        relevance_percentage = sum(1 for r in results if r['is_relevant']) / total_samples * 100
        error_percentage = sum(1 for r in results if r['generated_answer'].startswith('Error:')) / total_samples * 100
        
        summary = {
            "total_samples": total_samples,
            "average_answer_length": round(avg_answer_length, 2),
            "legal_terms_coverage": round(legal_terms_percentage, 2),
            "relevance_score": round(relevance_percentage, 2),
            "error_rate": round(error_percentage, 2),
            "success_rate": round(100 - error_percentage, 2)
        }
        
        return summary
    
    def _save_evaluation_results(self, results: List[Dict], summary: Dict):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(LOGS_DIR, f"evaluation_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_file = os.path.join(LOGS_DIR, f"evaluation_summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Create readable report
        report_file = os.path.join(LOGS_DIR, f"evaluation_report_{timestamp}.txt")
        self._create_readable_report(results, summary, report_file)
        
        logger.info(f"Evaluation results saved to {LOGS_DIR}")
    
    def _create_readable_report(self, results: List[Dict], summary: Dict, report_file: str):
        """Create a human-readable evaluation report."""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LEGAL LLM EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary section
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            for key, value in summary.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            # Sample results
            f.write("SAMPLE EVALUATION RESULTS\n")
            f.write("-" * 40 + "\n")
            
            for i, result in enumerate(results[:5], 1):  # Show first 5 results
                f.write(f"\nSample {i}:\n")
                f.write(f"Question: {result['question']}\n")
                f.write(f"Generated Answer: {result['generated_answer'][:200]}...\n")
                f.write(f"Contains Legal Terms: {result['contains_legal_terms']}\n")
                f.write(f"Is Relevant: {result['is_relevant']}\n")
                f.write("-" * 40 + "\n")

def main():
    """Main function for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Legal LLM")
    parser.add_argument("--model_path", type=str, help="Path to the fine-tuned model")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = LegalModelEvaluator(args.model_path)
    
    # Run evaluation
    results = evaluator.evaluate_model(args.num_samples)
    
    # Print summary
    print("\nEvaluation Summary:")
    print("=" * 50)
    for key, value in results["summary"].items():
        print(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()