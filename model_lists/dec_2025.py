"""
December 2025 Model Catalog

Complete collection of open-source LLMs organized by size and category.
Use the helper functions to get models by category or size range.

Example usage:
    from model_lists.dec2025 import get_all_models, get_models_by_category, CATEGORIES

    # Get all models
    all_models = get_all_models()

    # Get specific category
    coding_models = get_models_by_category("coding")

    # Get all categories
    categories = CATEGORIES
"""

# ==============================================================================
# SMALL MODELS: 1-3B Parameters
# ==============================================================================

SMALL_GENERAL = [
  {
    "id": "meta-llama/Llama-3.2-1B",
    "name": "Llama-3.2-1B",
    "size_gb": 2.5,
    "category": "general",
    "description": "Distilled from 8B/70B, on-device"
  },
  {
    "id": "meta-llama/Llama-3.2-1B-Instruct",
    "name": "Llama-3.2-1B-Instruct",
    "size_gb": 2.5,
    "category": "general",
    "description": "Chat-optimized for mobile"
  },
  {
    "id": "meta-llama/Llama-3.2-3B",
    "name": "Llama-3.2-3B",
    "size_gb": 6.4,
    "category": "general",
    "description": "Best small Llama for 128K context"
  },
  {
    "id": "meta-llama/Llama-3.2-3B-Instruct",
    "name": "Llama-3.2-3B-Instruct",
    "size_gb": 6.4,
    "category": "general",
    "description": "Edge deployment champion"
  },
  {
    "id": "Qwen/Qwen2.5-0.5B",
    "name": "Qwen2.5-0.5B",
    "size_gb": 1,
    "category": "general",
    "description": "Smallest practical LLM"
  },
  {
    "id": "Qwen/Qwen2.5-1.5B",
    "name": "Qwen2.5-1.5B",
    "size_gb": 3,
    "category": "general",
    "description": "128K context, 29+ languages"
  },
  {
    "id": "Qwen/Qwen2.5-3B",
    "name": "Qwen2.5-3B",
    "size_gb": 6,
    "category": "general",
    "description": "Strong multilingual support"
  },
  {
    "id": "Qwen/Qwen2.5-0.5B-Instruct",
    "name": "Qwen2.5-0.5B-Instruct",
    "size_gb": 1,
    "category": "general",
    "description": "Micro-assistant deployment"
  },
  {
    "id": "Qwen/Qwen2.5-1.5B-Instruct",
    "name": "Qwen2.5-1.5B-Instruct",
    "size_gb": 3,
    "category": "general",
    "description": "JSON/structured output"
  },
  {
    "id": "Qwen/Qwen2.5-3B-Instruct",
    "name": "Qwen2.5-3B-Instruct",
    "size_gb": 6,
    "category": "general",
    "description": "Tool calling support"
  },
  {
    "id": "google/gemma-2b",
    "name": "Gemma-2B",
    "size_gb": 5,
    "category": "general",
    "description": "Trained on 2T tokens"
  },
  {
    "id": "google/gemma-2-2b",
    "name": "Gemma-2-2B",
    "size_gb": 5,
    "category": "general",
    "description": "Improved architecture"
  },
  {
    "id": "google/gemma-2-2b-it",
    "name": "Gemma-2-2B-Instruct",
    "size_gb": 5,
    "category": "general",
    "description": "Consumer GPU optimized"
  },
  {
    "id": "microsoft/phi-2",
    "name": "Phi-2",
    "size_gb": 5.5,
    "category": "general",
    "description": "Exceptional reasoning for size"
  },
  {
    "id": "mistralai/Ministral-3-3B-Instruct-2512",
    "name": "Ministral-3B",
    "size_gb": 4,
    "category": "general",
    "description": "Browser-runnable, vision capable"
  },
  {
    "id": "nvidia/OpenReasoning-Nemotron-1.5B",
    "name": "OpenReasoning-Nemotron-1.5B",
    "size_gb": 3,
    "category": "general",
    "description": "Edge reasoning model"
  },
  {
    "id": "allenai/OLMo-2-0425-1B",
    "name": "OLMo-2-1B",
    "size_gb": 2,
    "category": "general",
    "description": "Fully transparent training"
  },
  {
    "id": "bigcode/starcoder2-3b",
    "name": "StarCoder2-3B",
    "size_gb": 6,
    "category": "coding",
    "description": "Code generation specialist"
  },
  {
    "id": "google/paligemma-3b-pt-224",
    "name": "PaliGemma-3B",
    "size_gb": 6,
    "category": "multimodal",
    "description": "Vision-language, VQA"
  },
  {
    "id": "ibm-granite/granite-4.0-micro",
    "name": "Granite-4.0-Micro",
    "size_gb": 6,
    "category": "general",
    "description": "Enterprise-grade, PII-safe"
  },
  {
    "id": "state-spaces/mamba-2.8b",
    "name": "Mamba-2.8B",
    "size_gb": 5.5,
    "category": "general",
    "description": "Linear-time SSM, no attention"
  },
  {
    "id": "RWKV/v6-Finch-1B6-HF",
    "name": "RWKV-6-World-1.6B",
    "size_gb": 3.2,
    "category": "general",
    "description": "Infinite context, constant memory"
  }
]

SMALL_CODING = [
  {
    "id": "Qwen/Qwen2.5-Coder-0.5B",
    "name": "Qwen2.5-Coder-0.5B",
    "size_gb": 1,
    "category": "coding",
    "description": "Smallest code model"
  },
  {
    "id": "Qwen/Qwen2.5-Coder-1.5B",
    "name": "Qwen2.5-Coder-1.5B",
    "size_gb": 3,
    "category": "coding",
    "description": "40+ languages, FIM support"
  },
  {
    "id": "Qwen/Qwen2.5-Coder-3B",
    "name": "Qwen2.5-Coder-3B",
    "size_gb": 6,
    "category": "coding",
    "description": "IDE assistant ready"
  },
  {
    "id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "name": "Qwen2.5-Coder-1.5B-Instruct",
    "size_gb": 3,
    "category": "coding",
    "description": "Code chat optimized"
  },
  {
    "id": "deepseek-ai/deepseek-coder-1.3b-base",
    "name": "DeepSeek-Coder-1.3B",
    "size_gb": 2.6,
    "category": "coding",
    "description": "86 programming languages"
  },
  {
    "id": "google/codegemma-2b",
    "name": "CodeGemma-2B",
    "size_gb": 5,
    "category": "coding",
    "description": "Fill-in-middle objective"
  }
]

SMALL_MATH = [
  {
    "id": "Qwen/Qwen2.5-Math-1.5B",
    "name": "Qwen2.5-Math-1.5B",
    "size_gb": 3,
    "category": "math",
    "description": "Chain-of-thought reasoning"
  },
  {
    "id": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "name": "Qwen2.5-Math-1.5B-Instruct",
    "size_gb": 3,
    "category": "math",
    "description": "Tool-integrated reasoning"
  }
]

# ==============================================================================
# MEDIUM MODELS: 7-14B Parameters
# ==============================================================================

MEDIUM_GENERAL = [
  {
    "id": "meta-llama/Meta-Llama-3-8B",
    "name": "Llama-3-8B",
    "size_gb": 16,
    "category": "general",
    "description": "128K tokenizer, 15T tokens"
  },
  {
    "id": "meta-llama/Meta-Llama-3-8B-Instruct",
    "name": "Llama-3-8B-Instruct",
    "size_gb": 16,
    "category": "general",
    "description": "RLHF-aligned chat"
  },
  {
    "id": "meta-llama/Llama-3.1-8B",
    "name": "Llama-3.1-8B",
    "size_gb": 16,
    "category": "general",
    "description": "128K context, multilingual"
  },
  {
    "id": "meta-llama/Llama-3.1-8B-Instruct",
    "name": "Llama-3.1-8B-Instruct",
    "size_gb": 16,
    "category": "general",
    "description": "Tool use built-in"
  },
  {
    "id": "Qwen/Qwen2.5-7B",
    "name": "Qwen2.5-7B",
    "size_gb": 14,
    "category": "general",
    "description": "18T token training"
  },
  {
    "id": "Qwen/Qwen2.5-7B-Instruct",
    "name": "Qwen2.5-7B-Instruct",
    "size_gb": 14,
    "category": "general",
    "description": "Role-play, JSON output"
  },
  {
    "id": "Qwen/Qwen2.5-14B",
    "name": "Qwen2.5-14B",
    "size_gb": 28,
    "category": "general",
    "description": "Best mid-size base model"
  },
  {
    "id": "Qwen/Qwen2.5-14B-Instruct",
    "name": "Qwen2.5-14B-Instruct",
    "size_gb": 28,
    "category": "general",
    "description": "Excellent instruction following"
  },
  {
    "id": "google/gemma-7b",
    "name": "Gemma-7B",
    "size_gb": 17,
    "category": "general",
    "description": "Google quality, consumer friendly"
  },
  {
    "id": "google/gemma-2-9b",
    "name": "Gemma-2-9B",
    "size_gb": 18,
    "category": "general",
    "description": "Competitive with 2x larger models"
  },
  {
    "id": "google/gemma-2-9b-it",
    "name": "Gemma-2-9B-Instruct",
    "size_gb": 18,
    "category": "general",
    "description": "torch.compile 6x speedup"
  },
  {
    "id": "microsoft/phi-4",
    "name": "Phi-4",
    "size_gb": 28,
    "category": "reasoning",
    "description": "80.4% MATH, synthetic data"
  },
  {
    "id": "microsoft/Phi-4-reasoning",
    "name": "Phi-4-Reasoning",
    "size_gb": 28,
    "category": "reasoning",
    "description": "Deep reasoning specialist"
  },
  {
    "id": "mistralai/Mistral-7B-Instruct-v0.3",
    "name": "Mistral-7B-Instruct-v0.3",
    "size_gb": 14,
    "category": "general",
    "description": "Function calling, tool use"
  },
  {
    "id": "mistralai/Mistral-Nemo-Instruct-2407",
    "name": "Mistral-Nemo-12B",
    "size_gb": 24.5,
    "category": "general",
    "description": "128K context, NVIDIA collab"
  },
  {
    "id": "mistralai/Ministral-3-8B-Instruct-2512",
    "name": "Ministral-8B",
    "size_gb": 9,
    "category": "general",
    "description": "Vision capable, edge ready"
  },
  {
    "id": "mistralai/Ministral-3-14B-Instruct-2512",
    "name": "Ministral-14B",
    "size_gb": 14,
    "category": "general",
    "description": "Vision + reasoning"
  },
  {
    "id": "nvidia/OpenReasoning-Nemotron-7B",
    "name": "OpenReasoning-Nemotron-7B",
    "size_gb": 14,
    "category": "reasoning",
    "description": "Reasoning distilled from R1"
  },
  {
    "id": "nvidia/OpenReasoning-Nemotron-14B",
    "name": "OpenReasoning-Nemotron-14B",
    "size_gb": 28,
    "category": "reasoning",
    "description": "SOTA at 14B for science"
  },
  {
    "id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "name": "Llama-3.1-Nemotron-Nano-8B",
    "size_gb": 16,
    "category": "general",
    "description": "RTX-ready reasoning"
  },
  {
    "id": "nvidia/Nemotron-H-8B-Base-8K",
    "name": "Nemotron-H-8B",
    "size_gb": 16,
    "category": "general",
    "description": "Hybrid Mamba-Transformer"
  },
  {
    "id": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "name": "NVIDIA-Nemotron-Nano-9B-v2",
    "size_gb": 18,
    "category": "general",
    "description": "3-6x faster than Qwen3-8B"
  },
  {
    "id": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base",
    "name": "NVIDIA-Nemotron-Nano-12B-v2",
    "size_gb": 24,
    "category": "general",
    "description": "Hybrid architecture, 128K"
  },
  {
    "id": "nvidia/Minitron-8B-Base",
    "name": "Minitron-8B",
    "size_gb": 16,
    "category": "general",
    "description": "Pruned Nemotron-4 15B"
  },
  {
    "id": "nvidia/Mistral-NeMo-Minitron-8B-Instruct",
    "name": "Mistral-NeMo-Minitron-8B",
    "size_gb": 16,
    "category": "general",
    "description": "RAG, function calling"
  },
  {
    "id": "tiiuae/falcon-11B",
    "name": "Falcon2-11B",
    "size_gb": 22,
    "category": "general",
    "description": "FlashAttention optimized"
  },
  {
    "id": "allenai/OLMo-2-1124-7B",
    "name": "OLMo-2-7B",
    "size_gb": 14,
    "category": "general",
    "description": "100% open (data, code, logs)"
  },
  {
    "id": "01-ai/Yi-6B",
    "name": "Yi-6B",
    "size_gb": 12,
    "category": "general",
    "description": "Strong bilingual EN/CN"
  },
  {
    "id": "internlm/internlm2_5-7b-chat",
    "name": "InternLM2.5-7B-Chat",
    "size_gb": 14,
    "category": "general",
    "description": "100+ web pages tool use"
  },
  {
    "id": "internlm/internlm3-8b-instruct",
    "name": "InternLM3-8B-Instruct",
    "size_gb": 16,
    "category": "general",
    "description": "Deep thinking mode"
  },
  {
    "id": "RWKV/rwkv-6-world-7b",
    "name": "RWKV-6-World-7B",
    "size_gb": 14,
    "category": "general",
    "description": "Linear complexity, no KV-cache"
  },
  {
    "id": "tiiuae/falcon-mamba-7b",
    "name": "FalconMamba-7B",
    "size_gb": 14,
    "category": "general",
    "description": "SSM architecture"
  },
  {
    "id": "tiiuae/Falcon-H1-7B-Instruct",
    "name": "Falcon-H1-7B",
    "size_gb": 14,
    "category": "general",
    "description": "Hybrid Transformer-SSM"
  }
]

MEDIUM_CODING = [
  {
    "id": "Qwen/Qwen2.5-Coder-7B",
    "name": "Qwen2.5-Coder-7B",
    "size_gb": 14,
    "category": "coding",
    "description": "5.5T code tokens trained"
  },
  {
    "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "name": "Qwen2.5-Coder-7B-Instruct",
    "size_gb": 14,
    "category": "coding",
    "description": "Code agent ready"
  },
  {
    "id": "Qwen/Qwen2.5-Coder-14B",
    "name": "Qwen2.5-Coder-14B",
    "size_gb": 28,
    "category": "coding",
    "description": "IDE integration, FIM"
  },
  {
    "id": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "name": "Qwen2.5-Coder-14B-Instruct",
    "size_gb": 28,
    "category": "coding",
    "description": "Professional coding assistant"
  },
  {
    "id": "deepseek-ai/deepseek-coder-6.7b-base",
    "name": "DeepSeek-Coder-6.7B",
    "size_gb": 13,
    "category": "coding",
    "description": "2T tokens, 87% code"
  },
  {
    "id": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "name": "DeepSeek-Coder-6.7B-Instruct",
    "size_gb": 13,
    "category": "coding",
    "description": "16K context"
  },
  {
    "id": "google/codegemma-7b",
    "name": "CodeGemma-7B",
    "size_gb": 17,
    "category": "coding",
    "description": "500B additional code tokens"
  },
  {
    "id": "google/codegemma-7b-it",
    "name": "CodeGemma-7B-Instruct",
    "size_gb": 17,
    "category": "coding",
    "description": "Natural language to code"
  },
  {
    "id": "bigcode/starcoder2-7b",
    "name": "StarCoder2-7B",
    "size_gb": 14,
    "category": "coding",
    "description": "600+ languages"
  },
  {
    "id": "bigcode/starcoder2-15b",
    "name": "StarCoder2-15B",
    "size_gb": 32,
    "category": "coding",
    "description": "4T tokens, SOTA code gen"
  }
]

MEDIUM_MATH = [
  {
    "id": "Qwen/Qwen2.5-Math-7B",
    "name": "Qwen2.5-Math-7B",
    "size_gb": 14,
    "category": "math",
    "description": "85.3% MATH with TIR"
  },
  {
    "id": "Qwen/Qwen2.5-Math-7B-Instruct",
    "name": "Qwen2.5-Math-7B-Instruct",
    "size_gb": 14,
    "category": "math",
    "description": "Chain-of-thought + tools"
  },
  {
    "id": "deepseek-ai/deepseek-math-7b-base",
    "name": "DeepSeek-Math-7B",
    "size_gb": 14,
    "category": "math",
    "description": "Theorem proving"
  }
]

MEDIUM_MULTIMODAL = [
  {
    "id": "meta-llama/Llama-3.2-11B-Vision",
    "name": "Llama-3.2-11B-Vision",
    "size_gb": 22,
    "category": "multimodal",
    "description": "Image + cross-attention"
  },
  {
    "id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "name": "Llama-3.2-11B-Vision-Instruct",
    "size_gb": 22,
    "category": "multimodal",
    "description": "VQA, DocVQA, captioning"
  },
  {
    "id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "name": "Qwen2.5-VL-7B-Instruct",
    "size_gb": 14,
    "category": "multimodal",
    "description": "1+ hour video understanding"
  },
  {
    "id": "Qwen/Qwen2.5-Omni-7B",
    "name": "Qwen2.5-Omni-7B",
    "size_gb": 14,
    "category": "multimodal",
    "description": "Text+image+audio+video to text+speech"
  },
  {
    "id": "mistralai/Pixtral-12B-2409",
    "name": "Pixtral-12B",
    "size_gb": 24,
    "category": "multimodal",
    "description": "Variable image resolution"
  },
  {
    "id": "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
    "name": "Llama-3.1-Nemotron-Nano-VL-8B",
    "size_gb": 16,
    "category": "multimodal",
    "description": "Document OCR, Jetson ready"
  },
  {
    "id": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    "name": "NVIDIA-Nemotron-Nano-VL-12B",
    "size_gb": 24,
    "category": "multimodal",
    "description": "#1 OCRBenchV2"
  },
  {
    "id": "microsoft/Phi-3.5-vision-instruct",
    "name": "Phi-3.5-Vision",
    "size_gb": 8.4,
    "category": "multimodal",
    "description": "Multi-image, 128K context"
  },
  {
    "id": "microsoft/Phi-4-multimodal-instruct",
    "name": "Phi-4-Multimodal",
    "size_gb": 11,
    "category": "multimodal",
    "description": "Speech + vision"
  }
]

MEDIUM_LONG_CONTEXT = [
  {
    "id": "Qwen/Qwen2.5-7B-Instruct-1M",
    "name": "Qwen2.5-7B-Instruct-1M",
    "size_gb": 14,
    "category": "general",
    "description": "1 million tokens"
  },
  {
    "id": "Qwen/Qwen2.5-14B-Instruct-1M",
    "name": "Qwen2.5-14B-Instruct-1M",
    "size_gb": 28,
    "category": "general",
    "description": "1 million tokens"
  }
]

# ==============================================================================
# LARGE MODELS: 30-34B Parameters
# ==============================================================================

LARGE_GENERAL = [
  {
    "id": "Qwen/Qwen2.5-32B",
    "name": "Qwen2.5-32B",
    "size_gb": 64,
    "category": "general",
    "description": "Excellent base for fine-tuning"
  },
  {
    "id": "Qwen/Qwen2.5-32B-Instruct",
    "name": "Qwen2.5-32B-Instruct",
    "size_gb": 64,
    "category": "general",
    "description": "Strong instruction following"
  },
  {
    "id": "Qwen/QwQ-32B",
    "name": "QwQ-32B",
    "size_gb": 65,
    "category": "reasoning",
    "description": "o1-like deep reasoning"
  },
  {
    "id": "google/gemma-2-27b",
    "name": "Gemma-2-27B",
    "size_gb": 54,
    "category": "general",
    "description": "Single GPU (A100/H100)"
  },
  {
    "id": "google/gemma-2-27b-it",
    "name": "Gemma-2-27B-Instruct",
    "size_gb": 54,
    "category": "general",
    "description": "13T token training"
  },
  {
    "id": "google/gemma-3-27b-it",
    "name": "Gemma-3-27B-Instruct",
    "size_gb": 54,
    "category": "general",
    "description": "128K context, multimodal"
  },
  {
    "id": "nvidia/OpenReasoning-Nemotron-32B",
    "name": "OpenReasoning-Nemotron-32B",
    "size_gb": 64,
    "category": "reasoning",
    "description": "SOTA math/code/science"
  },
  {
    "id": "allenai/Olmo-3-1125-32B",
    "name": "OLMo-3-32B",
    "size_gb": 64,
    "category": "general",
    "description": "Fully transparent training"
  },
  {
    "id": "01-ai/Yi-34B",
    "name": "Yi-34B",
    "size_gb": 68,
    "category": "general",
    "description": "Top Hugging Face leaderboard"
  },
  {
    "id": "01-ai/Yi-34B-200K",
    "name": "Yi-34B-200K",
    "size_gb": 68,
    "category": "general",
    "description": "200K context"
  },
  {
    "id": "01-ai/Yi-1.5-34B",
    "name": "Yi-1.5-34B",
    "size_gb": 68,
    "category": "general",
    "description": "Commercial friendly"
  },
  {
    "id": "CohereForAI/c4ai-command-r-v01",
    "name": "Command-R",
    "size_gb": 70,
    "category": "general",
    "description": "RAG-optimized, 128K ctx"
  }
]

LARGE_CODING = [
  {
    "id": "Qwen/Qwen2.5-Coder-32B",
    "name": "Qwen2.5-Coder-32B",
    "size_gb": 64,
    "category": "coding",
    "description": "Matches GPT-4o on coding"
  },
  {
    "id": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "name": "Qwen2.5-Coder-32B-Instruct",
    "size_gb": 64,
    "category": "coding",
    "description": "Best open-source coder"
  },
  {
    "id": "deepseek-ai/deepseek-coder-33b-base",
    "name": "DeepSeek-Coder-33B",
    "size_gb": 66,
    "category": "coding",
    "description": "86 languages, FIM"
  },
  {
    "id": "deepseek-ai/deepseek-coder-33b-instruct",
    "name": "DeepSeek-Coder-33B-Instruct",
    "size_gb": 66,
    "category": "coding",
    "description": "Legendary coding model"
  },
  {
    "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "name": "DeepSeek-R1-Distill-Qwen-32B",
    "size_gb": 64,
    "category": "coding",
    "description": "Reasoning distilled"
  }
]

LARGE_MULTIMODAL = [
  {
    "id": "Qwen/Qwen2.5-VL-32B-Instruct",
    "name": "Qwen2.5-VL-32B-Instruct",
    "size_gb": 64,
    "category": "multimodal",
    "description": "Document + video understanding"
  },
  {
    "id": "01-ai/Yi-VL-34B",
    "name": "Yi-VL-34B",
    "size_gb": 68,
    "category": "multimodal",
    "description": "Strong bilingual vision"
  }
]

# ==============================================================================
# EXTRA LARGE MODELS: 40-72B Parameters
# ==============================================================================

XLARGE_GENERAL = [
  {
    "id": "meta-llama/Meta-Llama-3-70B",
    "name": "Llama-3-70B",
    "size_gb": 140,
    "category": "general",
    "description": "Foundation model"
  },
  {
    "id": "meta-llama/Llama-3.1-70B",
    "name": "Llama-3.1-70B",
    "size_gb": 140,
    "category": "general",
    "description": "128K context, tool use"
  },
  {
    "id": "meta-llama/Llama-3.1-70B-Instruct",
    "name": "Llama-3.1-70B-Instruct",
    "size_gb": 140,
    "category": "general",
    "description": "Production-ready chat"
  },
  {
    "id": "meta-llama/Llama-3.3-70B-Instruct",
    "name": "Llama-3.3-70B-Instruct",
    "size_gb": 140,
    "category": "general",
    "description": "405B performance at 70B cost"
  },
  {
    "id": "Qwen/Qwen2.5-72B",
    "name": "Qwen2.5-72B",
    "size_gb": 144,
    "category": "general",
    "description": "Best Qwen base model"
  },
  {
    "id": "Qwen/Qwen2.5-72B-Instruct",
    "name": "Qwen2.5-72B-Instruct",
    "size_gb": 144,
    "category": "general",
    "description": "18T token training"
  },
  {
    "id": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "name": "Llama-3.1-Nemotron-70B-Instruct",
    "size_gb": 140,
    "category": "general",
    "description": "SOTA Arena Hard, AlpacaEval"
  },
  {
    "id": "nvidia/Llama-3.1-Nemotron-70B-Reward-HF",
    "name": "Llama-3.1-Nemotron-70B-Reward",
    "size_gb": 140,
    "category": "reward",
    "description": "#1 RewardBench"
  },
  {
    "id": "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    "name": "Llama-3.3-Nemotron-Super-49B",
    "size_gb": 98,
    "category": "general",
    "description": "Single H200 GPU deployment"
  },
  {
    "id": "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    "name": "Llama-3.3-Nemotron-Super-49B-v1.5",
    "size_gb": 98,
    "category": "general",
    "description": "Improved agentic reasoning"
  },
  {
    "id": "nvidia/Nemotron-H-56B-Base",
    "name": "Nemotron-H-56B",
    "size_gb": 112,
    "category": "general",
    "description": "Hybrid Mamba-2, 20T tokens"
  },
  {
    "id": "nvidia/Nemotron-H-47B-Base",
    "name": "Nemotron-H-47B",
    "size_gb": 94,
    "category": "general",
    "description": "20% faster, ~1M context FP4"
  },
  {
    "id": "tiiuae/falcon-40b",
    "name": "Falcon-40B",
    "size_gb": 80,
    "category": "general",
    "description": "RefinedWeb trained"
  },
  {
    "id": "ai21labs/Jamba-v0.1",
    "name": "Jamba-v0.1",
    "size_gb": 100,
    "category": "general",
    "description": "SSM-Transformer hybrid"
  },
  {
    "id": "ai21labs/AI21-Jamba-1.5-Mini",
    "name": "Jamba-1.5-Mini",
    "size_gb": 100,
    "category": "general",
    "description": "256K context, 2.5x faster"
  },
  {
    "id": "ai21labs/AI21-Jamba-Mini-1.6",
    "name": "Jamba-1.6-Mini",
    "size_gb": 100,
    "category": "general",
    "description": "Function calling, JSON mode"
  }
]

XLARGE_MATH = [
  {
    "id": "Qwen/Qwen2.5-Math-72B",
    "name": "Qwen2.5-Math-72B",
    "size_gb": 144,
    "category": "math",
    "description": "87.8% MATH with TIR"
  },
  {
    "id": "Qwen/Qwen2.5-Math-72B-Instruct",
    "name": "Qwen2.5-Math-72B-Instruct",
    "size_gb": 144,
    "category": "math",
    "description": "Best open-source math"
  },
  {
    "id": "Qwen/Qwen2.5-Math-RM-72B",
    "name": "Qwen2.5-Math-RM-72B",
    "size_gb": 144,
    "category": "reward",
    "description": "Reward model for RL"
  }
]

XLARGE_MULTIMODAL = [
  {
    "id": "meta-llama/Llama-3.2-90B-Vision",
    "name": "Llama-3.2-90B-Vision",
    "size_gb": 180,
    "category": "multimodal",
    "description": "Largest open VLM"
  },
  {
    "id": "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "name": "Llama-3.2-90B-Vision-Instruct",
    "size_gb": 180,
    "category": "multimodal",
    "description": "Visual reasoning"
  },
  {
    "id": "Qwen/Qwen2.5-VL-72B-Instruct",
    "name": "Qwen2.5-VL-72B-Instruct",
    "size_gb": 144,
    "category": "multimodal",
    "description": "1+ hour video, visual agent"
  },
  {
    "id": "Qwen/QVQ-72B-Preview",
    "name": "QVQ-72B",
    "size_gb": 144,
    "category": "multimodal",
    "description": "Visual reasoning, 70.3% MMMU"
  }
]

XLARGE_MOE = [
  {
    "id": "mistralai/Mixtral-8x7B-v0.1",
    "name": "Mixtral-8x7B",
    "size_gb": 90,
    "category": "general",
    "description": "70B quality, 13B compute"
  },
  {
    "id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "name": "Mixtral-8x7B-Instruct",
    "size_gb": 90,
    "category": "general",
    "description": "Efficient MoE chat"
  }
]

# ==============================================================================
# MASSIVE MODELS: 100B+ Parameters
# ==============================================================================

MASSIVE_DENSE = [
  {
    "id": "meta-llama/Llama-3.1-405B",
    "name": "Llama-3.1-405B",
    "size_gb": 810,
    "category": "general",
    "description": "Largest open dense model"
  },
  {
    "id": "meta-llama/Llama-3.1-405B-Instruct",
    "name": "Llama-3.1-405B-Instruct",
    "size_gb": 810,
    "category": "general",
    "description": "Synthetic data generation"
  },
  {
    "id": "meta-llama/Llama-3.1-405B-Instruct-FP8",
    "name": "Llama-3.1-405B-Instruct-FP8",
    "size_gb": 405,
    "category": "general",
    "description": "Half memory, 8xH100"
  },
  {
    "id": "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
    "name": "Llama-3.1-Nemotron-Ultra-253B",
    "size_gb": 475,
    "category": "general",
    "description": "Single 8xH100 node, reasoning toggle"
  },
  {
    "id": "CohereForAI/c4ai-command-r-plus",
    "name": "Command-R+",
    "size_gb": 208,
    "category": "general",
    "description": "RAG champion, 128K context"
  },
  {
    "id": "CohereForAI/c4ai-command-r-plus-4bit",
    "name": "Command-R+-4bit",
    "size_gb": 52,
    "category": "general",
    "description": "Memory-efficient RAG"
  },
  {
    "id": "tiiuae/falcon-180B",
    "name": "Falcon-180B",
    "size_gb": 360,
    "category": "general",
    "description": "RefinedWeb, 3.5T tokens"
  }
]

MASSIVE_MOE = [
  {
    "id": "mistralai/Mixtral-8x22B-v0.1",
    "name": "Mixtral-8x22B",
    "size_gb": 282,
    "category": "general",
    "description": "64K context, top-2 routing"
  },
  {
    "id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "name": "Mixtral-8x22B-Instruct",
    "size_gb": 282,
    "category": "general",
    "description": "Efficient large-scale"
  },
  {
    "id": "deepseek-ai/DeepSeek-V3",
    "name": "DeepSeek-V3",
    "size_gb": 685,
    "category": "frontier",
    "description": "93.3% KV-cache reduction, MTP"
  },
  {
    "id": "deepseek-ai/DeepSeek-V3.1",
    "name": "DeepSeek-V3.1",
    "size_gb": 685,
    "category": "frontier",
    "description": "Hybrid thinking mode"
  },
  {
    "id": "deepseek-ai/DeepSeek-V3.2",
    "name": "DeepSeek-V3.2",
    "size_gb": 685,
    "category": "frontier",
    "description": "Sparse attention"
  },
  {
    "id": "deepseek-ai/DeepSeek-R1",
    "name": "DeepSeek-R1",
    "size_gb": 685,
    "category": "frontier",
    "description": "o1-competitive reasoning"
  },
  {
    "id": "deepseek-ai/DeepSeek-V2",
    "name": "DeepSeek-V2",
    "size_gb": 472,
    "category": "general",
    "description": "MLA architecture"
  },
  {
    "id": "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "name": "DeepSeek-Coder-V2-236B",
    "size_gb": 472,
    "category": "coding",
    "description": "338 languages, GPT-4 level"
  },
  {
    "id": "deepseek-ai/DeepSeek-Math-V2",
    "name": "DeepSeek-Math-V2",
    "size_gb": 685,
    "category": "math",
    "description": "IMO gold-level math"
  },
  {
    "id": "mistralai/Mistral-Large-3-675B-Base-2512",
    "name": "Mistral-Large-3-Base",
    "size_gb": 1350,
    "category": "frontier",
    "description": "Granular MoE + vision"
  },
  {
    "id": "mistralai/Mistral-Large-3-675B-Instruct-2512",
    "name": "Mistral-Large-3-Instruct",
    "size_gb": 689,
    "category": "frontier",
    "description": "#2 LMArena OSS"
  },
  {
    "id": "nvidia/Nemotron-4-340B-Base",
    "name": "Nemotron-4-340B-Base",
    "size_gb": 635,
    "category": "general",
    "description": "9T tokens, 50+ languages"
  },
  {
    "id": "nvidia/Nemotron-4-340B-Instruct",
    "name": "Nemotron-4-340B-Instruct",
    "size_gb": 635,
    "category": "general",
    "description": "98% synthetic data alignment"
  },
  {
    "id": "nvidia/Nemotron-4-340B-Reward",
    "name": "Nemotron-4-340B-Reward",
    "size_gb": 635,
    "category": "reward",
    "description": "Top RewardBench"
  },
  {
    "id": "ai21labs/AI21-Jamba-1.5-Large",
    "name": "Jamba-1.5-Large",
    "size_gb": 750,
    "category": "general",
    "description": "256K context, hybrid SSM"
  },
  {
    "id": "ai21labs/AI21-Jamba-Large-1.6",
    "name": "Jamba-1.6-Large",
    "size_gb": 750,
    "category": "general",
    "description": "Production-ready hybrid"
  },
  {
    "id": "mistralai/Pixtral-Large-Instruct-2411",
    "name": "Pixtral-Large",
    "size_gb": 300,
    "category": "multimodal",
    "description": "Frontier multimodal"
  }
]

# ==============================================================================
# SAFETY AND GUARD MODELS
# ==============================================================================

SAFETY_MODELS = [
  {
    "id": "meta-llama/Llama-Guard-3-8B",
    "name": "Llama-Guard-3-8B",
    "size_gb": 16,
    "category": "safety",
    "description": "MLCommons taxonomy, 8 languages"
  },
  {
    "id": "meta-llama/Llama-Guard-3-8B-INT8",
    "name": "Llama-Guard-3-8B-INT8",
    "size_gb": 8,
    "category": "safety",
    "description": "Quantized deployment"
  },
  {
    "id": "meta-llama/Llama-Guard-3-1B",
    "name": "Llama-Guard-3-1B",
    "size_gb": 2,
    "category": "safety",
    "description": "Edge safety classification"
  },
  {
    "id": "meta-llama/Llama-Guard-3-11B-Vision",
    "name": "Llama-Guard-3-11B-Vision",
    "size_gb": 22,
    "category": "safety",
    "description": "First open multimodal safety"
  },
  {
    "id": "meta-llama/Prompt-Guard-86M",
    "name": "Prompt-Guard-86M",
    "size_gb": 0.2,
    "category": "safety",
    "description": "Injection/jailbreak detection"
  },
  {
    "id": "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3",
    "name": "Nemotron-Safety-Guard-8B-v3",
    "size_gb": 16,
    "category": "safety",
    "description": "9 languages, 23 categories"
  },
  {
    "id": "nvidia/Nemotron-Content-Safety-Reasoning-4B",
    "name": "Nemotron-Content-Safety-4B",
    "size_gb": 8,
    "category": "safety",
    "description": "Dynamic policy guardrail"
  },
  {
    "id": "nvidia/llama-3.1-nemoguard-8b-content-safety",
    "name": "NemoGuard-8B-Content-Safety",
    "size_gb": 16,
    "category": "safety",
    "description": "LoRA adapter, topic-following"
  }
]

# ==============================================================================
# HELPER FUNCTIONS AND EXPORTS
# ==============================================================================

# Map of all model collections
ALL_MODEL_COLLECTIONS = {
  # Small models (1-3B)
  "small_general": SMALL_GENERAL,
  "small_coding": SMALL_CODING,
  "small_math": SMALL_MATH,

  # Medium models (7-14B)
  "medium_general": MEDIUM_GENERAL,
  "medium_coding": MEDIUM_CODING,
  "medium_math": MEDIUM_MATH,
  "medium_multimodal": MEDIUM_MULTIMODAL,
  "medium_long_context": MEDIUM_LONG_CONTEXT,

  # Large models (30-34B)
  "large_general": LARGE_GENERAL,
  "large_coding": LARGE_CODING,
  "large_multimodal": LARGE_MULTIMODAL,

  # Extra large models (40-72B)
  "xlarge_general": XLARGE_GENERAL,
  "xlarge_math": XLARGE_MATH,
  "xlarge_multimodal": XLARGE_MULTIMODAL,
  "xlarge_moe": XLARGE_MOE,

  # Massive models (100B+)
  "massive_dense": MASSIVE_DENSE,
  "massive_moe": MASSIVE_MOE,

  # Specialized
  "safety": SAFETY_MODELS,
}

# Category mapping
CATEGORIES = {
  "general": "General Purpose",
  "coding": "Code Generation",
  "math": "Mathematical Reasoning",
  "reasoning": "Deep Reasoning",
  "multimodal": "Vision/Audio/Video",
  "safety": "Safety & Guardrails",
  "reward": "Reward Models",
  "frontier": "Frontier Models",
}

# Size categories
SIZE_CATEGORIES = {
  "small": "1-3B Parameters",
  "medium": "7-14B Parameters",
  "large": "30-34B Parameters",
  "xlarge": "40-72B Parameters",
  "massive": "100B+ Parameters",
}


def get_all_models():
  """Get all models from all collections."""
  all_models = []
  for collection in ALL_MODEL_COLLECTIONS.values():
    all_models.extend(collection)
  return all_models


def get_models_by_category(category: str):
  """
  Get all models for a specific category.

  Args:
    category: Category name (e.g., "coding", "general", "multimodal")

  Returns:
    List of models matching the category
  """
  all_models = get_all_models()
  return [m for m in all_models if m["category"] == category]


def get_models_by_size(size_category: str):
  """
  Get all models for a specific size category.

  Args:
    size_category: Size category ("small", "medium", "large", "xlarge", "massive")

  Returns:
    List of models in that size range
  """
  all_models = []
  for collection_name, collection in ALL_MODEL_COLLECTIONS.items():
    if collection_name.startswith(size_category):
        all_models.extend(collection)
  return all_models


def get_collection(collection_name: str):
  """
  Get a specific model collection by name.

  Args:
    collection_name: Collection name (e.g., "small_coding", "large_general")

  Returns:
    List of models in that collection
  """
  return ALL_MODEL_COLLECTIONS.get(collection_name, [])

def print_catalog_summary():
  """Print a summary of all available models."""
  all_models = get_all_models()
  total_size = sum(m["size_gb"] for m in all_models)

  print(f"\n{'='*80}")
  print(f"DECEMBER 2025 MODEL CATALOG")
  print(f"{'='*80}")
  print(f"Total models: {len(all_models)}")
  print(f"Total size: ~{total_size:,.0f} GB")
  print(f"\nCategories: {len(CATEGORIES)}")

  for cat, name in sorted(CATEGORIES.items()):
    cat_models = get_models_by_category(cat)
    cat_size = sum(m["size_gb"] for m in cat_models)
    print(f"  {name:25} {len(cat_models):3} models  ~{cat_size:6,.0f} GB")

  print(f"\nSize Ranges:")
  for size_key, size_name in SIZE_CATEGORIES.items():
    size_models = get_models_by_size(size_key)
    size_total = sum(m["size_gb"] for m in size_models)
    print(f"  {size_name:25} {len(size_models):3} models  ~{size_total:6,.0f} GB")

  print(f"{'='*80}\n")


if __name__ == "__main__":
  print_catalog_summary()