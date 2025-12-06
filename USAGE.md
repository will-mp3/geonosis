# Geonosis Droid Factory - Usage Guide

Simple script to download LLM models from Hugging Face.

## Quick Start

```bash
# Set your Hugging Face token
export HF_TOKEN="hf_your_token_here"

# Download a specific model
python3 droid_factory.py --models "Qwen2.5-72B-Instruct"
```

## Configuration

Set these environment variables (or add to your `.bashrc`/`.zshrc`):

```bash
export HF_TOKEN="hf_your_token_here"          # Required
export OUTPUT_BASE="/Volumes/robots/models"   # Required
```

Or use a `.env` file (requires `python-dotenv`):
```bash
HF_TOKEN=hf_your_token_here
OUTPUT_BASE=/Volumes/robots/models
```

## Usage Examples

### Interactive Mode

Browse and select from all available models:
```bash
python3 droid_factory.py
```

### Download Specific Models

```bash
# Single model
python3 droid_factory.py --models "Qwen2.5-72B-Instruct"

# Multiple models
python3 droid_factory.py --models "Phi-4" "QwQ-32B" "DeepSeek-Coder-6.7B"
```

### Download by Category

```bash
# All coding models
python3 droid_factory.py --category coding

# All reasoning models
python3 droid_factory.py --category reasoning

# All math models
python3 droid_factory.py --category math
```

### Download by Size

```bash
# Small models (1-3B parameters)
python3 droid_factory.py --size small

# Medium models (7-14B parameters)
python3 droid_factory.py --size medium

# Large models (30-34B parameters)
python3 droid_factory.py --size large
```

### List Available Options

```bash
# List all categories
python3 droid_factory.py --list-categories

# List all size ranges
python3 droid_factory.py --list-sizes
```

### Non-Interactive Mode

Skip confirmation prompts (useful for scripts):
```bash
python3 droid_factory.py --category coding --non-interactive
```

## Available Categories

- `general` - General purpose models
- `coding` - Code generation models
- `math` - Mathematical reasoning models
- `reasoning` - Deep reasoning models
- `multimodal` - Vision/audio/video models
- `safety` - Safety and guardrail models
- `reward` - Reward models
- `frontier` - Frontier models

## Available Sizes

- `small` - 1-3B parameters
- `medium` - 7-14B parameters
- `large` - 30-34B parameters
- `xlarge` - 40-72B parameters
- `massive` - 100B+ parameters

## Model Organization

Models are stored in:
```
$OUTPUT_BASE/
  ├── Qwen2.5-72B-Instruct/
  ├── Phi-4/
  ├── DeepSeek-Coder-6.7B/
  └── ...
```

Each model directory contains:
- `*.safetensors` - Model weights
- `*.json` - Configuration files
- `tokenizer*` - Tokenizer files
- `LICENSE` - License information

## Adding New Models

Edit `model_lists/master_list.py` to add new models. Follow the existing format:

```python
{
  "id": "org/model-name",           # HuggingFace repo ID
  "name": "Model-Name",              # Display name
  "size_gb": 144,                    # Approximate size in GB
  "category": "general",             # Category
  "description": "Brief description" # Description
}
```

## Monthly Model Lists

Create new monthly lists in `model_lists/`:
- `dec_2025.py` - December 2025 models
- `jan_2026.py` - January 2026 models (future)
- etc.

## Tips

- **Resume downloads**: Interrupted downloads automatically resume
- **Disk space**: Make sure you have enough space before downloading large models
- **Network**: Large models take time - use `caffeinate` to prevent sleep:
  ```bash
  caffeinate -i python3 droid_factory.py --models "Qwen2.5-72B-Instruct"
  ```
- **File formats**: Script downloads only safetensors (skips older .bin/.pt formats)

## Troubleshooting

**Authentication errors:**
```bash
# Check your token is set
echo $HF_TOKEN

# Get a token from: https://huggingface.co/settings/tokens
```

**Model not found:**
```bash
# List all available models interactively
python3 droid_factory.py

# Or check the catalog
python3 -m model_lists.master_list
```

**Out of space:**
```bash
# Check available space
df -h $OUTPUT_BASE

# Download to different location
export OUTPUT_BASE="/path/with/more/space"
```

## Help

```bash
python3 droid_factory.py --help
```
