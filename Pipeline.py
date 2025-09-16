# main.py
"""Main execution script for the Indian Legal Corpus LLM project."""

import argparse
import logging
import os
import sys
from typing import Optional
from ui_utils import toast, loading

# Import our modules
from data_loader import DataLoader
from fine_tuner import LegalLMFineTuner
from inference_engine import LegalQuestionAnswerer
from evaluator import LegalModelEvaluator
from project_structure import MODELS_DIR, DATA_DIR, LOGS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalLMPipeline:
    """Complete pipeline for Legal LLM training and inference."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.fine_tuner = LegalLMFineTuner()
        self.qa_system = None
        self.evaluator = None
    
    def prepare_data(self, source: str = "huggingface"):
        """Prepare data for training."""
        logger.info("Starting data preparation...")
        with loading("Preparing data..."):
            dataset_dict = self.data_loader.load_and_prepare_data(source=source)
        toast("Data preparation completed", type="success")
        logger.info("Data preparation completed successfully!")
        return dataset_dict
    
    def train_model(self, dataset_dict=None, source: str = "huggingface"):
        """Train the model."""
        logger.info("Starting model training...")
        if dataset_dict is None:
            dataset_dict = self.prepare_data(source)
        with loading("Training model..."):
            model_path = self.fine_tuner.train(dataset_dict)
        with loading("Merging and saving model..."):
            merged_path = self.fine_tuner.merge_and_save_model(model_path)
        toast(f"Training completed. Saved to: {merged_path}", type="success")
        logger.info(f"Model training completed! Final model saved to: {merged_path}")
        return merged_path
    
    def run_inference(self, model_path: Optional[str] = None, interactive: bool = True):
        """Run inference with the trained model."""
        logger.info("Starting inference...")
        with loading("Loading inference engine..."):
            self.qa_system = LegalQuestionAnswerer(model_path)
            self.qa_system.load()
        toast("Inference engine ready", type="success")
        
        if interactive:
            # Start interactive chat
            print("\n" + "="*60)
            print("LEGAL AI ASSISTANT - INTERACTIVE MODE")
            print("="*60)
            print("Ask legal questions or type 'quit' to exit.")
            print("-"*60)
            
            while True:
                try:
                    question = input("\nYour question: ").strip()
                    
                    if question.lower() in ['quit', 'exit', 'q']:
                        print("Thank you for using Legal AI Assistant!")
                        break
                    
                    if not question:
                        continue
                    
                    print("\nGenerating response...")
                    with loading("Thinking..."):
                        result = self.qa_system.answer(question)
                    print(f"\nAnswer: {result['answer']}")
                    print("-"*60)
                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    toast(f"Error: {e}", type="error")
        
        return self.qa_system
    
    def evaluate_model(self, model_path: Optional[str] = None, num_samples: int = 10):
        """Evaluate the model performance."""
        logger.info("Starting model evaluation...")
        with loading("Running evaluation..."):
            self.evaluator = LegalModelEvaluator(model_path)
            results = self.evaluator.evaluate_model(num_samples)
        toast("Evaluation completed", type="success")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        summary = results["summary"]
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("\nDetailed results saved to logs directory.")
        return results
    
    def run_full_pipeline(self, source: str = "huggingface", eval_samples: int = 10):
        """Run the complete pipeline: data preparation, training, and evaluation."""
        logger.info("Starting full pipeline...")
        
        try:
            # Step 1: Prepare data
            dataset_dict = self.prepare_data(source)
            
            # Step 2: Train model
            model_path = self.train_model(dataset_dict)
            
            # Step 3: Evaluate model
            self.evaluate_model(model_path, eval_samples)
            
            # Step 4: Start interactive inference
            self.run_inference(model_path, interactive=True)
            
        except Exception as e:
            toast(f"Pipeline failed: {e}", type="error")
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Indian Legal Corpus LLM Training and Inference")
    
    parser.add_argument(
        "--mode",
        choices=["data", "train", "inference", "evaluate", "full", "ui"],
        default="full",
        help="Operation mode"
    )
    
    parser.add_argument(
        "--data_source",
        choices=["huggingface", "csv"],
        default="huggingface",
        help="Data source for training"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained model for inference/evaluation"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        help="Single question for inference mode"
    )
    
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=10,
        help="Number of samples for evaluation"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat for inference"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = LegalLMPipeline()
    
    try:
        if args.mode == "data":
            # Data preparation only
            pipeline.prepare_data(args.data_source)
            
        elif args.mode == "train":
            # Training only
            pipeline.train_model(source=args.data_source)
            
        elif args.mode == "inference":
            # Inference only
            qa_system = pipeline.run_inference(args.model_path, args.interactive)
            
            if args.question:
                result = qa_system.answer(args.question)
                print(f"Question: {args.question}")
                print(f"Answer: {result['answer']}")
            
        elif args.mode == "evaluate":
            # Evaluation only
            pipeline.evaluate_model(args.model_path, args.eval_samples)
            
        elif args.mode == "full":
            # Full pipeline
            pipeline.run_full_pipeline(args.data_source, args.eval_samples)
            
        elif args.mode == "ui":
            # Launch Streamlit UI
            print("Launching Streamlit UI...")
            os.system("streamlit run streamlit_app.py")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        toast(f"Error: {e}", type="error")
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# requirements.txt content
REQUIREMENTS = """torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
peft>=0.6.0
accelerate>=0.24.0
bitsandbytes>=0.41.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
streamlit>=1.28.0
huggingface_hub>=0.17.0
tokenizers>=0.14.0
sentencepiece>=0.1.99
tqdm>=4.65.0
"""

# Save requirements.txt
def create_requirements_file():
    """Create requirements.txt file."""
    with open("requirements.txt", "w") as f:
        f.write(REQUIREMENTS.strip())
    print("requirements.txt created successfully!")

if __name__ == "__main__":
    # Create requirements file if running directly
    create_requirements_file()
    main()