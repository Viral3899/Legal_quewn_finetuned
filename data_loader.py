# data_loader.py
"""Data loading and preprocessing utilities for Indian Legal Corpus."""

import pandas as pd
import json
import os
import re
from typing import List, Dict, Tuple
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
import logging
from project_structure import DATA_CONFIG, DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of Indian legal corpus data."""
    
    def __init__(self):
        self.data_config = DATA_CONFIG
        
    def load_from_huggingface(self, dataset_name: str = None) -> pd.DataFrame:
        """Load dataset from HuggingFace."""
        dataset_name = dataset_name or self.data_config.dataset_name
        
        try:
            logger.info(f"Loading dataset from HuggingFace: {dataset_name}")
            dataset = load_dataset(dataset_name)
            
            # Convert to pandas DataFrame
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                # If no train split, use the first available split
                split_name = list(dataset.keys())[0]
                df = dataset[split_name].to_pandas()
                
            logger.info(f"Successfully loaded {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset from HuggingFace: {e}")
            raise
    
    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        try:
            logger.info(f"Loading dataset from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"Successfully loaded {len(df)} records from CSV")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal terminology
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        
        # Strip and return
        return text.strip()
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataframe."""
        logger.info("Starting data preprocessing...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Identify potential question and answer columns
        columns = df_clean.columns.tolist()
        logger.info(f"Available columns: {columns}")
        
        # Try to identify question and answer columns
        question_col = None
        answer_col = None
        
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['question', 'query', 'prompt', 'input']):
                question_col = col
            elif any(keyword in col_lower for keyword in ['answer', 'response', 'output', 'completion']):
                answer_col = col
        
        # If not found, use first two columns
        if question_col is None:
            question_col = columns[0] if len(columns) > 0 else None
        if answer_col is None:
            answer_col = columns[1] if len(columns) > 1 else columns[0]
        
        logger.info(f"Using question column: {question_col}, answer column: {answer_col}")
        
        # Clean the data
        if question_col and question_col in df_clean.columns:
            df_clean['question'] = df_clean[question_col].apply(self.clean_text)
        else:
            raise ValueError("No suitable question column found")
        
        if answer_col and answer_col in df_clean.columns:
            df_clean['answer'] = df_clean[answer_col].apply(self.clean_text)
        else:
            raise ValueError("No suitable answer column found")
        
        # Remove empty rows
        df_clean = df_clean[(df_clean['question'] != "") & (df_clean['answer'] != "")]
        
        # Reset index
        df_clean = df_clean.reset_index(drop=True)
        
        logger.info(f"Data preprocessing completed. {len(df_clean)} records remaining.")
        return df_clean[['question', 'answer']]
    
    def format_for_training(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Format data for training with proper prompt structure."""
        formatted_data = []
        
        for _, row in df.iterrows():
            # Create a proper prompt format for legal Q&A
            prompt = f"<|im_start|>system\nYou are a helpful assistant specializing in Indian legal matters. Provide accurate and informative answers to legal questions.<|im_end|>\n<|im_start|>user\n{row['question']}<|im_end|>\n<|im_start|>assistant\n"
            completion = f"{row['answer']}<|im_end|>"
            
            formatted_data.append({
                "prompt": prompt,
                "completion": completion,
                "text": prompt + completion
            })
        
        return formatted_data
    
    def create_splits(self, data: List[Dict[str, str]]) -> DatasetDict:
        """Create train/validation/test splits."""
        # First split: separate test set
        train_val_data, test_data = train_test_split(
            data, 
            test_size=self.data_config.test_split, 
            random_state=42
        )
        
        # Second split: separate train and validation
        val_size = self.data_config.val_split / (1 - self.data_config.test_split)
        train_data, val_data = train_test_split(
            train_val_data, 
            test_size=val_size, 
            random_state=42
        )
        
        # Convert to HuggingFace datasets
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })
        
        logger.info(f"Dataset splits created:")
        logger.info(f"  Train: {len(train_data)} samples")
        logger.info(f"  Validation: {len(val_data)} samples")
        logger.info(f"  Test: {len(test_data)} samples")
        
        return dataset_dict
    
    def save_to_jsonl(self, data: List[Dict[str, str]], filename: str):
        """Save data to JSONL format."""
        filepath = os.path.join(DATA_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Data saved to {filepath}")
    
    def load_and_prepare_data(self, source: str = "huggingface") -> DatasetDict:
        """Main method to load and prepare data for training."""
        
        # Load data
        if source == "huggingface":
            df = self.load_from_huggingface()
        elif source == "csv":
            csv_path = os.path.join(DATA_DIR, "legal_data.csv")
            df = self.load_from_csv(csv_path)
        else:
            raise ValueError("Source must be 'huggingface' or 'csv'")
        
        # Preprocess data
        df_clean = self.preprocess_data(df)
        
        # Format for training
        formatted_data = self.format_for_training(df_clean)
        
        # Save to JSONL
        self.save_to_jsonl(formatted_data, "legal_corpus_formatted.jsonl")
        
        # Create splits
        dataset_dict = self.create_splits(formatted_data)
        
        # Save splits
        self.save_to_jsonl(dataset_dict['train'], "train_data.jsonl")
        self.save_to_jsonl(dataset_dict['validation'], "val_data.jsonl")
        self.save_to_jsonl(dataset_dict['test'], "test_data.jsonl")
        
        return dataset_dict

if __name__ == "__main__":
    loader = DataLoader()
    dataset = loader.load_and_prepare_data(source="huggingface")
    print("Data loading and preparation completed!")