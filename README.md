# Indian Legal Corpus LLM – Fine-tuning, Inference, and Evaluation

A practical pipeline to fine-tune Qwen models on Indian legal Q&A using LoRA, run inference (CLI and Streamlit UI), and evaluate results. This README matches the current code layout and filenames in this repository.

## Project Structure

```
.
├─ data/                         # JSONL datasets (formatted + splits)
├─ logs/                         # Evaluation reports and summaries
├─ models/                       # Checkpoints and merged models
├─ data_loader.py                # Load, clean, split, and format data
├─ fine_tuner.py                 # LoRA fine-tuning and merge utilities
├─ inference_engine.py           # Inference and Q&A wrapper
├─ evaluator.py                  # Evaluation suite and report writer
├─ legal_knowledge_base.py       # IPC facts fallback
├─ model_checker.py              # Scan/load/test best available model
├─ quick_test.py                 # Minimal smoke test for loading + QA
├─ streamlit_app.py              # Web UI (chat, training, evaluation)
├─ project_structure.py          # Configs and directory paths
├─ main_script.py                # CLI entrypoint for full pipeline
└─ README.md                     # This file
```

## Requirements

Python 3.9+ is recommended.

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
. .venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers datasets peft accelerate bitsandbytes scikit-learn pandas numpy streamlit huggingface_hub tokenizers sentencepiece tqdm
```

If you have limited resources or CPU-only, training will still run with conservative defaults.

## Configuration

Key settings (model, training, data, LoRA, and paths) live in `project_structure.py` as dataclasses:
- `MODEL_CONFIG`: base model name, max length, sampling params
- `TRAINING_CONFIG`: epochs, batch sizes, lr, save/eval/log steps
- `DATA_CONFIG`: dataset name and split ratios
- `LORA_CONFIG`: rank, alpha, dropout, target modules
- Paths: `DATA_DIR`, `MODELS_DIR`, `LOGS_DIR`

Adjust these before running for custom behavior.

## Quick Start

- Run the full pipeline (prepare data → train → merge → evaluate → interactive chat):
```bash
python main_script.py --mode full --eval_samples 10
```

- Launch the Streamlit UI:
```bash
python main_script.py --mode ui
# or directly
streamlit run streamlit_app.py
```

## CLI Usage (main_script.py)

```bash
# Data preparation (HuggingFace by default)
python main_script.py --mode data --data_source huggingface

# Or use your own CSV located at data/legal_data.csv
python main_script.py --mode data --data_source csv

# Train with LoRA (uses config in project_structure.py)
python main_script.py --mode train --data_source huggingface

# Evaluate a model
python main_script.py --mode evaluate --model_path models/legal_qwen_merged --eval_samples 20

# Inference: interactive chat
python main_script.py --mode inference --interactive --model_path models/legal_qwen_merged

# Inference: single question
python main_script.py --mode inference --question "What is Section 302 of IPC?" --model_path models/legal_qwen_merged
```

Notes:
- If `--model_path` is omitted, the code attempts to discover a merged model under `models/`.
- LoRA checkpoints are stored under timestamped folders in `models/` and later merged to `models/legal_qwen_merged`.

## Data Format

During preparation, the project writes JSONL files to `data/`:
- `legal_corpus_formatted.jsonl`: concatenated prompt+completion records used for LM training
- `train_data.jsonl`, `val_data.jsonl`, `test_data.jsonl`: split datasets

Each record contains keys: `prompt`, `completion`, and `text` (prompt+completion).

If using CSV, place `data/legal_data.csv` with columns that include question/answer semantics, e.g.:
```csv
question,answer
"What is Section 302 of IPC?","Section 302 deals with punishment for murder..."
```

## Training Details (fine_tuner.py)

- Loads the base model and tokenizer from `MODEL_CONFIG.base_model_name`
- Applies LoRA adapters per `LORA_CONFIG`
- Tokenizes using the concatenated `text` field
- Trains with `Trainer` using `TRAINING_CONFIG`
- Saves checkpoints under `models/legal_qwen_finetuned_YYYYMMDD_HHMMSS/`
- Merges LoRA into a standalone model at `models/legal_qwen_merged/`

Resource notes:
- Defaults are CPU-friendly; you can enable GPU by configuring your environment and adjusting dtypes/device map.

## Inference (inference_engine.py)

High-level interface: `LegalQuestionAnswerer`.
- Automatic fallback to `legal_knowledge_base.py` for common IPC sections (e.g., 302/304/307/375/420) and low-quality generations
- Example programmatic use:
```python
from inference_engine import LegalQuestionAnswerer
qa = LegalQuestionAnswerer("models/legal_qwen_merged")
qa.load()
print(qa.answer("What is IPC 302?")['answer'])
```

## Evaluation (evaluator.py)

- Loads test items from `data/test_data.jsonl` (or uses built-in samples if missing)
- Generates answers via `LegalQuestionAnswerer`
- Computes simple metrics: average answer length, legal-term coverage, relevance proxy, error rate
- Writes detailed JSON, summary JSON, and a text report under `logs/`

Run standalone:
```bash
python evaluator.py --model_path models/legal_qwen_merged --num_samples 10
```

## Streamlit UI (streamlit_app.py)

- Tabs for Chat, Training, and Evaluation
- Model selector prefers merged models under `models/`
- Chat history export and basic toasts for status

Run:
```bash
streamlit run streamlit_app.py
```

## Utilities

- `model_checker.py`: scan `models/`, pick best candidate, load, smoke-test, and optionally enter interactive Q&A
```bash
python model_checker.py            # full check
python model_checker.py -l         # list only
python model_checker.py -i         # include interactive test
```
- `quick_test.py`: minimal load + single QA demonstration

## Troubleshooting

- Out of memory or slow training: reduce `TRAINING_CONFIG.per_device_train_batch_size`, shorten `MODEL_CONFIG.max_length`, or use a smaller base model.
- Tokenizer pad token errors: the code sets `pad_token=eos_token` automatically.
- No models found at inference: ensure you trained and merged, or pass `--model_path` to a valid directory under `models/`.

## License

MIT. See repository license file if present.

## Acknowledgments

- Qwen models (Hugging Face: `Qwen`)
- LoRA (PEFT)
- Hugging Face Transformers/Datasets
- Streamlit
