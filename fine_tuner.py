# fine_tuner.py
"""Fine-tuning module for Qwen models on Indian Legal Corpus."""

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import DatasetDict
import logging
from datetime import datetime
from project_structure import MODEL_CONFIG, TRAINING_CONFIG, LORA_CONFIG, MODELS_DIR, LOGS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalLMFineTuner:
    """Fine-tuner for Legal Language Models using LoRA."""
    
    def __init__(self):
        self.model_config = MODEL_CONFIG
        self.training_config = TRAINING_CONFIG
        self.lora_config = LORA_CONFIG
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.model_config.base_model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.base_model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",  # Force CPU usage
                trust_remote_code=True
            )
            
            # Prepare model for training
            self.model = prepare_model_for_kbit_training(self.model)
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {e}")
            raise
    
    def setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning."""
        logger.info("Setting up LoRA configuration...")
        
        # Create LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias=self.lora_config.bias
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("LoRA setup completed")
    
    def tokenize_function(self, examples):
        """Tokenize the input examples."""
        # Tokenize the text
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.model_config.max_length,
            return_tensors=None
        )
        
        # Set labels (for causal LM, labels are the same as input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_datasets(self, dataset_dict: DatasetDict):
        """Prepare datasets for training."""
        logger.info("Preparing datasets for training...")
        
        # Tokenize datasets
        tokenized_datasets = dataset_dict.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
            desc="Tokenizing"
        )
        
        return tokenized_datasets
    
    def setup_trainer(self, tokenized_datasets):
        """Setup the trainer for fine-tuning."""
        logger.info("Setting up trainer...")
        
        # Create timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(MODELS_DIR, f"legal_qwen_finetuned_{timestamp}")
            
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            warmup_steps=self.training_config.warmup_steps,
            max_grad_norm=self.training_config.max_grad_norm,
            fp16=self.training_config.fp16,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            remove_unused_columns=self.training_config.remove_unused_columns,
            logging_dir=os.path.join(LOGS_DIR, f"logs_{timestamp}"),
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None  # Disable wandb/tensorboard for simplicity
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're doing causal LM, not masked LM
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        logger.info(f"Trainer setup completed. Output directory: {output_dir}")
        return output_dir
    
    def train(self, dataset_dict: DatasetDict):
        """Main training function."""
        logger.info("Starting fine-tuning process...")
        
        try:
            # Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Setup LoRA
            self.setup_lora()
            
            # Prepare datasets
            tokenized_datasets = self.prepare_datasets(dataset_dict)
            
            # Setup trainer
            output_dir = self.setup_trainer(tokenized_datasets)
            
            # Start training
            logger.info("Beginning training...")
            train_result = self.trainer.train()
            
            # Save the final model
            logger.info("Saving final model...")
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Save training metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            
            logger.info("Training completed successfully!")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def merge_and_save_model(self, model_path: str, merged_model_path: str = None):
        """Merge LoRA weights with base model and save."""
        if merged_model_path is None:
            merged_model_path = os.path.join(MODELS_DIR, "legal_qwen_merged")
        
        logger.info(f"Merging LoRA weights and saving to {merged_model_path}")
        
        try:
            # Load the fine-tuned model
            from peft import PeftModel
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Load LoRA model
            model = PeftModel.from_pretrained(base_model, model_path)
            
            # Merge weights
            merged_model = model.merge_and_unload()
            
            # Save merged model
            merged_model.save_pretrained(merged_model_path)
            
            # Save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.save_pretrained(merged_model_path)
            
            logger.info(f"Merged model saved to {merged_model_path}")
            return merged_model_path
            
        except Exception as e:
            logger.error(f"Error merging model: {e}")
            raise

def main():
    """Main function for training."""
    from data_loader import DataLoader
    
    # Load and prepare data
    data_loader = DataLoader()
    dataset_dict = data_loader.load_and_prepare_data()
    
    # Initialize fine-tuner
    fine_tuner = LegalLMFineTuner()
    
    # Train the model
    model_path = fine_tuner.train(dataset_dict)
    
    # Merge and save the final model
    merged_path = fine_tuner.merge_and_save_model(model_path)
    
    print(f"Training completed! Merged model saved to: {merged_path}")

if __name__ == "__main__":
    main()