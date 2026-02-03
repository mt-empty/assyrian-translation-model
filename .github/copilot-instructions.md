# AI Agent Instructions for Assyrian Translation Model

## Project Overview

This is a machine translation model for English to Assyrian/Eastern Syriac (Classical Syriac), fine-tuned from Helsinki-NLP's English-to-Arabic model (`Helsinki-NLP/opus-mt-en-ar`). The model targets Northeastern Neo-Aramaic languages but currently translates primarily Classical Syriac.

**Core Components:**
- [model.py](../model.py) - Main training pipeline (359 lines)
- [config.py](../config.py) - HuggingFace dataset configuration for EN-AS pairs
- [calc_vocab.py](../calc_vocab.py) - Vocabulary size calculation utility
- [test.py](../test.py) - Inference examples using the trained model

## Architecture & Key Decisions

### Transfer Learning Approach
The project uses transfer learning from Arabic (a related Semitic language) rather than training from scratch:
- **Base model**: `Helsinki-NLP/opus-mt-en-ar` (English → Arabic)
- **Why**: Leverages linguistic similarities between Arabic and Syriac scripts/morphology
- The model is adapted via fine-tuning on Assyrian/Syriac parallel corpora

### Tokenization Strategy
Uses **SentencePiece** with separate source/target vocabularies plus a shared vocabulary:
- English vocab size: 7,861 tokens
- Assyrian vocab size: 6,289 tokens (80% of English, calculated via [calc_vocab.py](../calc_vocab.py))
- Shared vocab: Combined 14,150 tokens
- Implementation: `MarianTokenizer` wraps SentencePiece models (`source.model`, `target.model`)

**Critical files generated during training:**
```
source.model, source.vocab  # English tokenizer
target.model, target.vocab  # Assyrian tokenizer
shared.model, shared.vocab  # Combined vocabulary
shared_vocab.json          # JSON format for MarianTokenizer (includes <pad> token)
```

### Dataset Structure
Two primary sources in [dataset/](../dataset/):
- `bible_data.csv` - Classical Syriac Bible parallel corpus
- `pericopes_data.csv` - Additional liturgical texts

Dataset processing flow in [model.py:41-56](../model.py):
1. CSV data → separate text files (`en-as.en`, `en-as.as`)
2. Loaded via custom [config.py](../config.py) dataset builder (HuggingFace datasets format)
3. 80/20 train/validation split with seed=20

## Training Workflow

### Running Training
```bash
python model.py  # Trains for 50 epochs by default
```

**Configuration constants** ([model.py:21-29](../model.py)):
```python
EPOCHS = 50
LEARNING_RATE = 5e-4
BATCH_SIZE = 180
ENGLISH_VOCAB_SIZE = 7861
ASSYRIAN_VOCAB_SIZE = 6289  # 80% of English
```

### Training Pipeline (Accelerate-based)
The model uses HuggingFace Accelerate for distributed training:
- Optimizer: AdamW with linear scheduler
- Evaluation: SacreBleu metric computed per epoch
- Checkpointing: Saves model + tokenizer after each epoch with timestamps

**Critical**: Model training requires:
- PyTorch with CUDA support (GPU recommended)
- Accelerate configured via `accelerate config` (if using multi-GPU)
- Uses `device = torch.device("cuda")` by default

### Inference/Testing
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("mt-empty/english-assyrian")
model = AutoModelForSeq2SeqLM.from_pretrained("mt-empty/english-assyrian")
translator = pipeline("translation", model=model, tokenizer=tokenizer)

print(translator("tomorrow morning"))
```
See [test.py](../test.py) for comprehensive examples.

## Development Conventions

### Code Style & Pre-commit Hooks
Project uses **pre-commit** hooks with Black and isort:
- Config: [py-hooks-config.toml](../py-hooks-config.toml)
- Line length: 88 characters (Black default)
- **Before contributing**: Run `pre-commit install` then `pre-commit run --all-files`

### Environment Setup
Two virtual environments in repo:
- `env/` - Linux environment (Python 3.8)
- `winenv/` - Windows environment

**Dependencies**: See [requirements.txt](../requirements.txt)
- Core: `transformers`, `datasets`, `sentencepiece`, `torch`, `accelerate`
- Metrics: `sacrebleu`
- Data: `pandas`, `nltk`

Optional: [other/setup.sh](../other/setup.sh) uses Spack for Python + CUDA builds

### File Encoding
All text files use **UTF-8 encoding** (critical for Syriac script):
```python
open(file, encoding="utf-8")  # Always specify encoding
```

## Important Quirks & Gotchas

1. **Unidirectional Translation**: Base model only translates EN→AR, not bidirectional. Attempting reverse translation fails ([model.py:113-119](../model.py)).

2. **Vocab Size Calculation**: [calc_vocab.py](../calc_vocab.py) uses NLTK stopwords removal + Unix tools (`tr`, `sort`, `uniq`) - only works on Linux.

3. **Padding Token**: `shared_vocab.json` manually adds `<pad>` token at index `SHARED_VOCAB_SIZE` ([model.py:93-94](../model.py)).

4. **SacreBleu Requirement**: Metric requires at least 4-gram inputs, returns 0 for shorter sequences.

5. **Dataset Column Names**: Custom dataset uses non-standard column name "of the Unleavened Bread" in [calc_vocab.py:20-22](../calc_vocab.py) - likely copy-paste artifact.

## Key File Relationships

```
model.py → config.py (dataset loader)
       ↓
   dataset/*.csv → en-as.{en,as} temp files → SentencePiece training
       ↓
   {source,target,shared}.{model,vocab} → MarianTokenizer → fine-tuning
       ↓
   model_en-as_*_epochs_*/ (saved checkpoints)
```

## Evaluation Metrics
- **SacreBleu score**: ~33 after 50 epochs (documented in [readme.md:25](../readme.md))
- Computed per epoch during training with validation set
- References require nested list format: `[[label.strip()]]`
