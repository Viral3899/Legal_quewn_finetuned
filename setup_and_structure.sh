#!/bin/bash
# setup.sh - Project Setup Script

echo "🚀 Setting up Indian Legal Corpus LLM Project..."

# Create project directory structure
echo "📁 Creating directory structure..."
mkdir -p data models logs

# Create Python virtual environment
echo "🐍 Setting up Python environment..."
python -m venv legal_llm_env

# Activate virtual environment (Linux/Mac)
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    source legal_llm_env/bin/activate
# Activate virtual environment (Windows)
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source legal_llm_env/Scripts/activate
fi

# Install requirements
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.35.0
pip install datasets>=2.14.0
pip install peft>=0.6.0
pip install accelerate>=0.24.0
pip install bitsandbytes>=0.41.0
pip install scikit-learn>=1.3.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install streamlit>=1.28.0
pip install huggingface_hub>=0.17.0
pip install tokenizers>=0.14.0
pip install sentencepiece>=0.1.99
pip install tqdm>=4.65.0

echo "✅ Setup completed!"
echo ""
echo "🎯 Quick Start Options:"
echo "1. Run full pipeline: python main.py --mode full"
echo "2. Launch web UI: python main.py --mode ui"
echo "3. Train only: python main.py --mode train"
echo "4. Interactive chat: python main.py --mode inference --interactive"
echo ""
echo "📖 See README.md for detailed instructions"

# Project structure template
cat > project_structure.txt << 'EOF'
indian-legal-llm/
├── 📄 config.py                    # Configuration settings
├── 📄 data_loader.py               # Data loading and preprocessing
├── 📄 fine_tuner.py               # Model fine-tuning with LoRA
├── 📄 inference.py                # Inference engine
├── 📄 evaluator.py                # Model evaluation
├── 📄 streamlit_app.py            # Web UI (Streamlit)
├── 📄 main.py                     # Main execution script
├── 📄 requirements.txt            # Dependencies
├── 📄 README.md                   # Documentation
├── 📄 setup.sh                    # Setup script
├── 📁 data/                       # Data directory
│   ├── legal_corpus_formatted.jsonl
│   ├── train_data.jsonl
│   ├── val_data.jsonl
│   └── test_data.jsonl
├── 📁 models/                     # Trained models
│   ├── legal_qwen_finetuned_*/
│   └── legal_qwen_merged/
├── 📁 logs/                       # Training and evaluation logs
│   ├── evaluation_results_*.json
│   ├── evaluation_summary_*.json
│   └── logs_*/
└── 📁 legal_llm_env/              # Virtual environment
EOF

echo "📁 Project structure saved to project_structure.txt"