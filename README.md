# Geonosis Droid Factory

A comprehensive collection of open-source large language models with automated download utilities and curated model documentation.

## Overview

This repository provides a monthly-updated catalog of state-of-the-art open-source language models and a Python-based download script for acquiring models from Hugging Face. The collection focuses on production-ready models across multiple categories including general-purpose, coding, reasoning, and specialized domains.

## Contents

### Model Documentation

`open_source_llm_master_list_december_2025.md` - Comprehensive catalog of 150+ actively maintained models, organized by parameter count (1B-670B+) with detailed specifications including:

- Exact Hugging Face repository IDs
- Model sizes and VRAM requirements  
- License classifications and commercial viability
- Benchmark performance metrics
- Architecture details and deployment guidance

Updated monthly to reflect the latest releases and performance data.

### Download Script

`download_models.py` - Automated download utility featuring:

- Interactive model selection interface
- Resume capability for interrupted downloads
- Progress tracking and error handling
- Automatic cache cleanup
- File format filtering to optimize storage
- Authentication support for gated models

## Requirements

- Python 3.11+
- `huggingface_hub` package
- Sufficient storage (models range from 1GB to 685GB)
- Hugging Face account and API token (for gated models)

## Installation
```bash
brew install huggingface_hub --break-system-packages
```

## Usage

### Configuration

Edit the script to set your output directory:
```python
OUTPUT_BASE = "/path/to/your/storage"
```

For gated models (Llama, etc.), add authentication:
```python
from huggingface_hub import login
login(token="your_hf_token_here")
```

### Running the Script
```bash
python download_models.py
```

The script presents an interactive menu to select models for download. Models can be downloaded individually or as a complete collection.

### Preventing System Sleep During Downloads
```bash
caffeinate -i python download_models.py
```

## Model Categories

The collection includes models optimized for:

- **General Purpose** - Instruction-following and chat applications
- **Coding** - Software development and code generation
- **Reasoning** - Mathematical and scientific problem-solving  
- **Multimodal** - Vision and audio understanding
- **Specialized** - Domain-specific applications (medical, legal, finance)

## Storage Considerations

- Models use safetensors format by default (more efficient than PyTorch)
- Automatic filtering excludes duplicate weight formats
- Cache cleanup prevents storage bloat
- Recommended filesystem: exFAT or APFS (supports large files >4GB)

## License

Model licenses vary by provider. Consult the model documentation for specific license terms. Common licenses include Apache 2.0, MIT, Llama Community License, and Qwen License.

## Updates

This repository is updated monthly to include:

- New model releases
- Updated benchmark scores
- Revised deployment recommendations
- Bug fixes and script improvements

## Contributing

Suggestions for model additions or script improvements can be submitted via issues or pull requests.

## Disclaimer

Models are provided by their respective organizations. This repository serves as a collection tool and reference guide. Users are responsible for compliance with individual model licenses and terms of service.