# config.py
"""Configuration settings for the Indian Legal Corpus LLM fine-tuning project."""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration settings."""
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Using smaller model for 4GB GPU
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

@dataclass
class TrainingConfig:
    """Training configuration settings."""
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # Reduced for 4GB GPU
    per_device_eval_batch_size: int = 1   # Reduced for 4GB GPU
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    fp16: bool = False  # Disabled for CPU training
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False

@dataclass
class DataConfig:
    """Data configuration settings."""
    dataset_name: str = "karan842/ipc-sections"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_prompt_length: int = 256
    max_response_length: int = 256

@dataclass
class LoRAConfig:
    """LoRA configuration for efficient fine-tuning."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Global configuration instances
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
DATA_CONFIG = DataConfig()
LORA_CONFIG = LoRAConfig()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)