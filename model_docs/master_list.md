# Open-Source LLM Master List for Hugging Face

**Over 150 actively maintained models** are catalogued here with exact Hugging Face repository IDs ready for `snapshot_download()`.

---

## Small models: 1-3B parameters

These models excel for edge deployment, mobile applications, and resource-constrained environments while maintaining surprising capability.

### General Purpose

| Model | Repo ID | Parameters | Size (BF16) | License | Key Strengths |
|-------|---------|------------|-------------|---------|---------------|
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B` | 1.24B | ~2.5 GB | Llama 3.2 | Distilled from 8B/70B, on-device |
| Llama 3.2 1B Instruct | `meta-llama/Llama-3.2-1B-Instruct` | 1.24B | ~2.5 GB | Llama 3.2 | Chat-optimized for mobile |
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B` | 3.21B | ~6.4 GB | Llama 3.2 | Best small Llama for 128K context |
| Llama 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` | 3.21B | ~6.4 GB | Llama 3.2 | Edge deployment champion |
| Qwen2.5 0.5B | `Qwen/Qwen2.5-0.5B` | 0.5B | ~1 GB | Apache 2.0 | Smallest practical LLM |
| Qwen2.5 1.5B | `Qwen/Qwen2.5-1.5B` | 1.5B | ~3 GB | Apache 2.0 | 128K context, 29+ languages |
| Qwen2.5 3B | `Qwen/Qwen2.5-3B` | 3B | ~6 GB | Qwen License | Strong multilingual support |
| Qwen2.5 0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ~1 GB | Apache 2.0 | Micro-assistant deployment |
| Qwen2.5 1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~3 GB | Apache 2.0 | JSON/structured output |
| Qwen2.5 3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` | 3B | ~6 GB | Qwen License | Tool calling support |
| Gemma 2B | `google/gemma-2b` | 2B | ~5 GB | Gemma | Trained on 2T tokens |
| Gemma 2 2B | `google/gemma-2-2b` | 2B | ~5 GB | Gemma | Improved architecture |
| Gemma 2 2B Instruct | `google/gemma-2-2b-it` | 2B | ~5 GB | Gemma | Consumer GPU optimized |
| Phi-2 | `microsoft/phi-2` | 2.7B | ~5.5 GB | MIT | Exceptional reasoning for size |
| Ministral 3B | `mistralai/Ministral-3-3B-Instruct-2512` | ~3B | ~4 GB | Apache 2.0 | Browser-runnable, vision capable |
| OpenReasoning-Nemotron 1.5B | `nvidia/OpenReasoning-Nemotron-1.5B` | 1.5B | ~3 GB | CC-BY-4.0 | Edge reasoning model |
| OLMo-2 1B | `allenai/OLMo-2-0425-1B` | 1B | ~2 GB | Apache 2.0 | Fully transparent training |
| StarCoder2 3B | `bigcode/starcoder2-3b` | 3B | ~6 GB | OpenRAIL-M | Code generation specialist |
| PaliGemma 3B | `google/paligemma-3b-pt-224` | 3B | ~6 GB | Gemma | Vision-language, VQA |
| Granite 4.0 Micro | `ibm-granite/granite-4.0-micro` | 3B | ~6 GB | Apache 2.0 | Enterprise-grade, PII-safe |
| Mamba 2.8B | `state-spaces/mamba-2.8b` | 2.77B | ~5.5 GB | Apache 2.0 | Linear-time SSM, no attention |
| RWKV-6 World 1.6B | `RWKV/v6-Finch-1B6-HF` | 1.6B | ~3.2 GB | Apache 2.0 | Infinite context, constant memory |

### Coding Specialists (1-3B)

| Model | Repo ID | Parameters | Size | License | Key Strengths |
|-------|---------|------------|------|---------|---------------|
| Qwen2.5-Coder 0.5B | `Qwen/Qwen2.5-Coder-0.5B` | 0.5B | ~1 GB | Apache 2.0 | Smallest code model |
| Qwen2.5-Coder 1.5B | `Qwen/Qwen2.5-Coder-1.5B` | 1.54B | ~3 GB | Apache 2.0 | 40+ languages, FIM support |
| Qwen2.5-Coder 3B | `Qwen/Qwen2.5-Coder-3B` | 3B | ~6 GB | Qwen-Research | IDE assistant ready |
| Qwen2.5-Coder 1.5B-Instruct | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.54B | ~3 GB | Apache 2.0 | Code chat optimized |
| DeepSeek-Coder 1.3B | `deepseek-ai/deepseek-coder-1.3b-base` | 1.3B | ~2.6 GB | DeepSeek | 86 programming languages |
| CodeGemma 2B | `google/codegemma-2b` | 2B | ~5 GB | Gemma | Fill-in-middle objective |

### Math Specialists (1-3B)

| Model | Repo ID | Parameters | Size | License | Key Strengths |
|-------|---------|------------|------|---------|---------------|
| Qwen2.5-Math 1.5B | `Qwen/Qwen2.5-Math-1.5B` | 1.5B | ~3 GB | Apache 2.0 | Chain-of-thought reasoning |
| Qwen2.5-Math 1.5B-Instruct | `Qwen/Qwen2.5-Math-1.5B-Instruct` | 1.5B | ~3 GB | Apache 2.0 | Tool-integrated reasoning |

---

## Medium models: 7-14B parameters

The sweet spot for single-GPU deployment on consumer hardware (RTX 4090, A10G) with strong general capabilities.

### General Purpose

| Model | Repo ID | Parameters | Size (BF16) | License | Key Strengths |
|-------|---------|------------|-------------|---------|---------------|
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B` | 8B | ~16 GB | Llama 3 | 128K tokenizer, 15T tokens |
| Llama 3 8B Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | ~16 GB | Llama 3 | RLHF-aligned chat |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B` | 8B | ~16 GB | Llama 3.1 | **128K context**, multilingual |
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 8B | ~16 GB | Llama 3.1 | Tool use built-in |
| Qwen2.5 7B | `Qwen/Qwen2.5-7B` | 7B | ~14 GB | Apache 2.0 | 18T token training |
| Qwen2.5 7B Instruct | `Qwen/Qwen2.5-7B-Instruct` | 7B | ~14 GB | Apache 2.0 | Role-play, JSON output |
| Qwen2.5 14B | `Qwen/Qwen2.5-14B` | 14B | ~28 GB | Apache 2.0 | Best mid-size base model |
| Qwen2.5 14B Instruct | `Qwen/Qwen2.5-14B-Instruct` | 14B | ~28 GB | Apache 2.0 | Excellent instruction following |
| Gemma 7B | `google/gemma-7b` | 7B | ~17 GB | Gemma | Google quality, consumer friendly |
| Gemma 2 9B | `google/gemma-2-9b` | 9B | ~18 GB | Gemma | **Competitive with 2x larger models** |
| Gemma 2 9B Instruct | `google/gemma-2-9b-it` | 9B | ~18 GB | Gemma | torch.compile 6x speedup |
| Phi-4 14B | `microsoft/phi-4` | 14B | ~28 GB | MIT | **80.4% MATH**, synthetic data |
| Phi-4 Reasoning | `microsoft/Phi-4-reasoning` | 14B | ~28 GB | MIT | Deep reasoning specialist |
| Mistral 7B v0.3 Instruct | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | ~14 GB | Apache 2.0 | Function calling, tool use |
| Mistral Nemo 12B | `mistralai/Mistral-Nemo-Instruct-2407` | 12B | ~24.5 GB | Apache 2.0 | 128K context, NVIDIA collab |
| Ministral 8B | `mistralai/Ministral-3-8B-Instruct-2512` | ~8B | ~9 GB (FP8) | Apache 2.0 | Vision capable, edge ready |
| Ministral 14B | `mistralai/Ministral-3-14B-Instruct-2512` | 14B | ~14 GB (FP8) | Apache 2.0 | Vision + reasoning |
| OpenReasoning-Nemotron 7B | `nvidia/OpenReasoning-Nemotron-7B` | 7B | ~14 GB | CC-BY-4.0 | Reasoning distilled from R1 |
| OpenReasoning-Nemotron 14B | `nvidia/OpenReasoning-Nemotron-14B` | 14B | ~28 GB | CC-BY-4.0 | SOTA at 14B for science |
| Llama-3.1-Nemotron-Nano 8B | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | 8B | ~16 GB | NVIDIA + Llama | **RTX-ready reasoning** |
| Nemotron-H 8B | `nvidia/Nemotron-H-8B-Base-8K` | 8B | ~16 GB | NVIDIA | Hybrid Mamba-Transformer |
| NVIDIA-Nemotron-Nano 9B v2 | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | 9B | ~18 GB | NVIDIA | **3-6x faster than Qwen3-8B** |
| NVIDIA-Nemotron-Nano 12B v2 | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base` | 12B | ~24 GB | NVIDIA | Hybrid architecture, 128K |
| Minitron 8B | `nvidia/Minitron-8B-Base` | 8B | ~16 GB | NVIDIA | Pruned Nemotron-4 15B |
| Mistral-NeMo-Minitron 8B | `nvidia/Mistral-NeMo-Minitron-8B-Instruct` | 8B | ~16 GB | NVIDIA | RAG, function calling |
| Falcon2 11B | `tiiuae/falcon-11B` | 11B | ~22 GB | Falcon 2.0 | FlashAttention optimized |
| OLMo-2 7B | `allenai/OLMo-2-1124-7B` | 7B | ~14 GB | Apache 2.0 | **100% open** (data, code, logs) |
| Yi-6B | `01-ai/Yi-6B` | 6B | ~12 GB | Yi License | Strong bilingual EN/CN |
| InternLM2.5 7B Chat | `internlm/internlm2_5-7b-chat` | 7B | ~14 GB | Custom | 100+ web pages tool use |
| InternLM3 8B Instruct | `internlm/internlm3-8b-instruct` | 8B | ~16 GB | Custom | Deep thinking mode |
| RWKV-6 World 7B | `RWKV/rwkv-6-world-7b` | 7B | ~14 GB | Apache 2.0 | Linear complexity, no KV-cache |
| FalconMamba 7B | `tiiuae/falcon-mamba-7b` | 7B | ~14 GB | Falcon-Mamba | SSM architecture |
| Falcon-H1 7B | `tiiuae/Falcon-H1-7B-Instruct` | 7B | ~14 GB | Apache 2.0 | Hybrid Transformer-SSM |

### Coding Specialists (7-14B)

| Model | Repo ID | Parameters | Size | License | Key Strengths |
|-------|---------|------------|------|---------|---------------|
| Qwen2.5-Coder 7B | `Qwen/Qwen2.5-Coder-7B` | 7B | ~14 GB | Apache 2.0 | 5.5T code tokens trained |
| Qwen2.5-Coder 7B Instruct | `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | ~14 GB | Apache 2.0 | Code agent ready |
| Qwen2.5-Coder 14B | `Qwen/Qwen2.5-Coder-14B` | 14B | ~28 GB | Apache 2.0 | IDE integration, FIM |
| Qwen2.5-Coder 14B Instruct | `Qwen/Qwen2.5-Coder-14B-Instruct` | 14B | ~28 GB | Apache 2.0 | Professional coding assistant |
| DeepSeek-Coder 6.7B | `deepseek-ai/deepseek-coder-6.7b-base` | 6.7B | ~13 GB | DeepSeek | 2T tokens, 87% code |
| DeepSeek-Coder 6.7B Instruct | `deepseek-ai/deepseek-coder-6.7b-instruct` | 6.7B | ~13 GB | DeepSeek | 16K context |
| CodeGemma 7B | `google/codegemma-7b` | 7B | ~17 GB | Gemma | 500B additional code tokens |
| CodeGemma 7B Instruct | `google/codegemma-7b-it` | 7B | ~17 GB | Gemma | Natural language to code |
| StarCoder2 7B | `bigcode/starcoder2-7b` | 7B | ~14 GB | OpenRAIL-M | 600+ languages |
| StarCoder2 15B | `bigcode/starcoder2-15b` | 15B | ~32 GB | OpenRAIL-M | **4T tokens, SOTA code gen** |

### Math Specialists (7-14B)

| Model | Repo ID | Parameters | Size | License | Key Strengths |
|-------|---------|------------|------|---------|---------------|
| Qwen2.5-Math 7B | `Qwen/Qwen2.5-Math-7B` | 7B | ~14 GB | Apache 2.0 | 85.3% MATH with TIR |
| Qwen2.5-Math 7B Instruct | `Qwen/Qwen2.5-Math-7B-Instruct` | 7B | ~14 GB | Apache 2.0 | Chain-of-thought + tools |
| DeepSeek-Math 7B | `deepseek-ai/deepseek-math-7b-base` | 7B | ~14 GB | DeepSeek | Theorem proving |

### Multimodal/Vision (7-14B)

| Model | Repo ID | Parameters | Size | License | Key Strengths |
|-------|---------|------------|------|---------|---------------|
| Llama 3.2 11B Vision | `meta-llama/Llama-3.2-11B-Vision` | 11B | ~22 GB | Llama 3.2 | Image + cross-attention |
| Llama 3.2 11B Vision Instruct | `meta-llama/Llama-3.2-11B-Vision-Instruct` | 11B | ~22 GB | Llama 3.2 | VQA, DocVQA, captioning |
| Qwen2.5-VL 7B Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | ~14 GB | Apache 2.0 | 1+ hour video understanding |
| Qwen2.5-Omni 7B | `Qwen/Qwen2.5-Omni-7B` | 7B | ~14 GB | Apache 2.0 | **Text+image+audio+video→text+speech** |
| Pixtral 12B | `mistralai/Pixtral-12B-2409` | 12.4B | ~24 GB | Apache 2.0 | Variable image resolution |
| Llama-3.1-Nemotron-Nano-VL 8B | `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` | ~8B | ~16 GB | NVIDIA + Llama | Document OCR, Jetson ready |
| NVIDIA-Nemotron-Nano-VL 12B | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | ~12B | ~24 GB | NVIDIA | **#1 OCRBenchV2** |
| Phi-3.5 Vision | `microsoft/Phi-3.5-vision-instruct` | 4.2B | ~8.4 GB | MIT | Multi-image, 128K context |
| Phi-4 Multimodal | `microsoft/Phi-4-multimodal-instruct` | 5.6B | ~11 GB | MIT | Speech + vision |

### Long Context Specialists (7-14B)

| Model | Repo ID | Parameters | Size | License | Context Length |
|-------|---------|------------|------|---------|----------------|
| Qwen2.5 7B Instruct 1M | `Qwen/Qwen2.5-7B-Instruct-1M` | 7B | ~14 GB | Apache 2.0 | **1 million tokens** |
| Qwen2.5 14B Instruct 1M | `Qwen/Qwen2.5-14B-Instruct-1M` | 14B | ~28 GB | Apache 2.0 | **1 million tokens** |

---

## Large models: 30-34B parameters

Ideal for multi-GPU servers, these models offer near-frontier performance for complex reasoning and generation tasks.

### General Purpose

| Model | Repo ID | Parameters | Size (BF16) | License | Key Strengths |
|-------|---------|------------|-------------|---------|---------------|
| Qwen2.5 32B | `Qwen/Qwen2.5-32B` | 32B | ~64 GB | Apache 2.0 | Excellent base for fine-tuning |
| Qwen2.5 32B Instruct | `Qwen/Qwen2.5-32B-Instruct` | 32B | ~64 GB | Apache 2.0 | Strong instruction following |
| QwQ 32B | `Qwen/QwQ-32B` | 32.5B | ~65 GB | Apache 2.0 | **o1-like deep reasoning** |
| Gemma 2 27B | `google/gemma-2-27b` | 27B | ~54 GB | Gemma | Single GPU (A100/H100) |
| Gemma 2 27B Instruct | `google/gemma-2-27b-it` | 27B | ~54 GB | Gemma | 13T token training |
| Gemma 3 27B Instruct | `google/gemma-3-27b-it` | 27B | ~54 GB | Gemma | **128K context, multimodal** |
| OpenReasoning-Nemotron 32B | `nvidia/OpenReasoning-Nemotron-32B` | 32B | ~64 GB | CC-BY-4.0 | **SOTA math/code/science** |
| OLMo-3 32B | `allenai/Olmo-3-1125-32B` | 32B | ~64 GB | Apache 2.0 | Fully transparent training |
| Yi-34B | `01-ai/Yi-34B` | 34B | ~68 GB | Yi License | Top Hugging Face leaderboard |
| Yi-34B 200K | `01-ai/Yi-34B-200K` | 34B | ~68 GB | Yi License | **200K context** |
| Yi-1.5 34B | `01-ai/Yi-1.5-34B` | 34B | ~68 GB | Apache 2.0 | Commercial friendly |
| Command R | `CohereForAI/c4ai-command-r-v01` | 35B | ~70 GB | CC-BY-NC | **RAG-optimized**, 128K ctx |

### Coding Specialists (30-34B)

| Model | Repo ID | Parameters | Size | License | Key Strengths |
|-------|---------|------------|------|---------|---------------|
| Qwen2.5-Coder 32B | `Qwen/Qwen2.5-Coder-32B` | 32B | ~64 GB | Apache 2.0 | **Matches GPT-4o on coding** |
| Qwen2.5-Coder 32B Instruct | `Qwen/Qwen2.5-Coder-32B-Instruct` | 32B | ~64 GB | Apache 2.0 | Best open-source coder |
| DeepSeek-Coder 33B | `deepseek-ai/deepseek-coder-33b-base` | 33B | ~66 GB | DeepSeek | 86 languages, FIM |
| DeepSeek-R1-Distill-Qwen 32B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | 32B | ~64 GB | MIT | Reasoning distilled |

### Multimodal (30-34B)

| Model | Repo ID | Parameters | Size | License | Key Strengths |
|-------|---------|------------|------|---------|---------------|
| Qwen2.5-VL 32B Instruct | `Qwen/Qwen2.5-VL-32B-Instruct` | 32B | ~64 GB | Apache 2.0 | Document + video understanding |
| Yi-VL 34B | `01-ai/Yi-VL-34B` | 34B | ~68 GB | Yi License | Strong bilingual vision |

---

## Extra large models: 40-72B parameters

High-performance models requiring significant compute (typically 2-4x A100/H100) but delivering near-frontier capabilities.

### General Purpose

| Model | Repo ID | Parameters | Size (BF16) | License | Key Strengths |
|-------|---------|------------|-------------|---------|---------------|
| Llama 3 70B | `meta-llama/Meta-Llama-3-70B` | 70B | ~140 GB | Llama 3 | Foundation model |
| **Llama 3.3 70B Instruct** | `meta-llama/Llama-3.3-70B-Instruct` | 70B | ~140 GB | Llama 3.3 | **405B performance at 70B cost** |
| Qwen2.5 72B | `Qwen/Qwen2.5-72B` | 72B | ~144 GB | Qwen License | Best Qwen base model |
| Qwen2.5 72B Instruct | `Qwen/Qwen2.5-72B-Instruct` | 72B | ~144 GB | Qwen License | 18T token training |
| Llama-3.1-Nemotron-70B-Instruct | `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` | 70B | ~140 GB | NVIDIA + Llama | **SOTA Arena Hard, AlpacaEval** |
| Llama-3.1-Nemotron-70B-Reward | `nvidia/Llama-3.1-Nemotron-70B-Reward-HF` | 70B | ~140 GB | NVIDIA + Llama | **#1 RewardBench** |
| Llama-3.3-Nemotron-Super 49B | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` | 49B | ~98 GB | NVIDIA + Llama | Single H200 GPU deployment |
| Llama-3.3-Nemotron-Super 49B v1.5 | `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | 49B | ~98 GB | NVIDIA + Llama | Improved agentic reasoning |
| Nemotron-H 56B Base | `nvidia/Nemotron-H-56B-Base` | 56B | ~112 GB | NVIDIA | **Hybrid Mamba-2**, 20T tokens |
| Nemotron-H 47B Base | `nvidia/Nemotron-H-47B-Base` | 47B | ~94 GB | NVIDIA | 20% faster, ~1M context FP4 |
| Falcon-40B | `tiiuae/falcon-40b` | 40B | ~80 GB | Apache 2.0 | RefinedWeb trained |
| Jamba v0.1 | `ai21labs/Jamba-v0.1` | 52B total/12B active | ~100 GB | Apache 2.0 | **SSM-Transformer hybrid** |
| Jamba 1.5 Mini | `ai21labs/AI21-Jamba-1.5-Mini` | 52B/12B active | ~100 GB | Jamba Open | 256K context, 2.5x faster |
| Jamba 1.6 Mini | `ai21labs/AI21-Jamba-Mini-1.6` | 52B/12B active | ~100 GB | Jamba Open | Function calling, JSON mode |

### Math Specialists (Extra Large)

| Model | Repo ID | Parameters | Size | License | Key Strengths |
|-------|---------|------------|------|---------|---------------|
| Qwen2.5-Math 72B | `Qwen/Qwen2.5-Math-72B` | 72B | ~144 GB | Qwen License | 87.8% MATH with TIR |
| Qwen2.5-Math 72B Instruct | `Qwen/Qwen2.5-Math-72B-Instruct` | 72B | ~144 GB | Qwen License | Best open-source math |
| Qwen2.5-Math-RM 72B | `Qwen/Qwen2.5-Math-RM-72B` | 72B | ~144 GB | Qwen License | Reward model for RL |

### Multimodal (Extra Large)

| Model | Repo ID | Parameters | Size | License | Key Strengths |
|-------|---------|------------|------|---------|---------------|
| Llama 3.2 90B Vision | `meta-llama/Llama-3.2-90B-Vision` | 90B | ~180 GB | Llama 3.2 | Largest open VLM |
| Llama 3.2 90B Vision Instruct | `meta-llama/Llama-3.2-90B-Vision-Instruct` | 90B | ~180 GB | Llama 3.2 | Visual reasoning |
| Qwen2.5-VL 72B Instruct | `Qwen/Qwen2.5-VL-72B-Instruct` | 72B | ~144 GB | Qwen License | **1+ hour video, visual agent** |
| QVQ 72B Preview | `Qwen/QVQ-72B-Preview` | 72B | ~144 GB | Qwen License | Visual reasoning, 70.3% MMMU |

### MoE Architectures (Extra Large)

| Model | Repo ID | Total/Active | Size | License | Key Strengths |
|-------|---------|--------------|------|---------|---------------|
| Mixtral 8x7B | `mistralai/Mixtral-8x7B-v0.1` | 46.7B/13B | ~90 GB | Apache 2.0 | 70B quality, 13B compute |
| Mixtral 8x7B Instruct | `mistralai/Mixtral-8x7B-Instruct-v0.1` | 46.7B/13B | ~90 GB | Apache 2.0 | Efficient MoE chat |

---

## Massive models: 100B+ parameters

Frontier-class models requiring substantial infrastructure (8+ H100 GPUs) for deployment.

### Dense Architectures

| Model | Repo ID | Parameters | Size (BF16) | License | Key Strengths |
|-------|---------|------------|-------------|---------|---------------|
| Llama 3.1 405B | `meta-llama/Llama-3.1-405B` | 405B | ~810 GB | Llama 3.1 | Largest open dense model |
| Llama 3.1 405B Instruct | `meta-llama/Llama-3.1-405B-Instruct` | 405B | ~810 GB | Llama 3.1 | Synthetic data generation |
| Llama 3.1 405B Instruct FP8 | `meta-llama/Llama-3.1-405B-Instruct-FP8` | 405B | ~405 GB | Llama 3.1 | Half memory, 8xH100 |
| Llama-3.1-Nemotron-Ultra 253B | `nvidia/Llama-3_1-Nemotron-Ultra-253B-v1` | 253B | ~475 GB (BF16) | NVIDIA + Llama | **Single 8xH100 node**, reasoning toggle |
| Command R+ | `CohereForAI/c4ai-command-r-plus` | 104B | ~208 GB | CC-BY-NC | **RAG champion**, 128K context |
| Command R+ 4bit | `CohereForAI/c4ai-command-r-plus-4bit` | 104B | ~52 GB | CC-BY-NC | Memory-efficient RAG |
| Falcon 180B | `tiiuae/falcon-180B` | 180B | ~360 GB | Falcon-180B TII | RefinedWeb, 3.5T tokens |

### MoE Architectures (Massive)

| Model | Repo ID | Total/Active | Size | License | Key Strengths |
|-------|---------|--------------|------|---------|---------------|
| Mixtral 8x22B | `mistralai/Mixtral-8x22B-v0.1` | 141B/39B | ~282 GB | Apache 2.0 | 64K context, top-2 routing |
| Mixtral 8x22B Instruct | `mistralai/Mixtral-8x22B-Instruct-v0.1` | 141B/39B | ~282 GB | Apache 2.0 | Efficient large-scale |
| **DeepSeek-V3** | `deepseek-ai/DeepSeek-V3` | 671B/37B | ~685 GB (FP8) | **MIT** | **93.3% KV-cache reduction**, MTP |
| DeepSeek-V3.1 | `deepseek-ai/DeepSeek-V3.1` | 671B/37B | ~685 GB | MIT | Hybrid thinking mode |
| DeepSeek-V3.2 | `deepseek-ai/DeepSeek-V3.2` | 671B/37B | ~685 GB | MIT | Sparse attention |
| **DeepSeek-R1** | `deepseek-ai/DeepSeek-R1` | 671B/37B | ~685 GB | **MIT** | **o1-competitive reasoning** |
| DeepSeek-V2 | `deepseek-ai/DeepSeek-V2` | 236B/21B | ~472 GB | DeepSeek | MLA architecture |
| DeepSeek-Coder-V2 236B | `deepseek-ai/DeepSeek-Coder-V2-Instruct` | 236B/21B | ~472 GB | DeepSeek | **338 languages**, GPT-4 level |
| DeepSeek-Math-V2 | `deepseek-ai/DeepSeek-Math-V2` | 671B/37B | ~685 GB | Apache 2.0 | IMO gold-level math |
| Mistral Large 3 Base | `mistralai/Mistral-Large-3-675B-Base-2512` | 675B/41B | ~1.35 TB | Apache 2.0 | Granular MoE + vision |
| Mistral Large 3 Instruct | `mistralai/Mistral-Large-3-675B-Instruct-2512` | 675B/41B | ~689 GB (FP8) | Apache 2.0 | **#2 LMArena OSS** |
| Nemotron-4 340B Base | `nvidia/Nemotron-4-340B-Base` | 340B | ~635 GB | NVIDIA | 9T tokens, 50+ languages |
| Nemotron-4 340B Instruct | `nvidia/Nemotron-4-340B-Instruct` | 340B | ~635 GB | NVIDIA | 98% synthetic data alignment |
| Nemotron-4 340B Reward | `nvidia/Nemotron-4-340B-Reward` | 340B | ~635 GB | NVIDIA | Top RewardBench |
| Jamba 1.5 Large | `ai21labs/AI21-Jamba-1.5-Large` | 398B/94B | ~750 GB | Jamba Open | **256K context**, hybrid SSM |
| Jamba 1.6 Large | `ai21labs/AI21-Jamba-Large-1.6` | 398B/94B | ~750 GB | Jamba Open | Production-ready hybrid |
| Pixtral Large | `mistralai/Pixtral-Large-Instruct-2411` | ~123B | ~300+ GB | Apache 2.0 | Frontier multimodal |

---

## Safety and guard models

Essential for production deployments requiring content moderation and safety filtering.

| Model | Repo ID | Parameters | Size | License | Key Strengths |
|-------|---------|------------|------|---------|---------------|
| Llama Guard 3 8B | `meta-llama/Llama-Guard-3-8B` | 8B | ~16 GB | Llama 3.1 | MLCommons taxonomy, 8 languages |
| Llama Guard 3 8B INT8 | `meta-llama/Llama-Guard-3-8B-INT8` | 8B | ~8 GB | Llama 3.1 | Quantized deployment |
| Llama Guard 3 1B | `meta-llama/Llama-Guard-3-1B` | 1B | ~2 GB | Llama 3.2 | Edge safety classification |
| Llama Guard 3 11B Vision | `meta-llama/Llama-Guard-3-11B-Vision` | 11B | ~22 GB | Llama 3.2 | **First open multimodal safety** |
| Prompt Guard 86M | `meta-llama/Prompt-Guard-86M` | 86M | ~0.2 GB | Llama | Injection/jailbreak detection |
| Nemotron Safety Guard 8B v3 | `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3` | 8B | ~16 GB | NVIDIA + Llama | 9 languages, 23 categories |
| Nemotron Content Safety 4B | `nvidia/Nemotron-Content-Safety-Reasoning-4B` | 4B | ~8 GB | NVIDIA | Dynamic policy guardrail |
| NemoGuard 8B Content Safety | `nvidia/llama-3.1-nemoguard-8b-content-safety` | 8B | ~16 GB | NVIDIA | LoRA adapter, topic-following |

---

## Alternative architecture highlights

Non-transformer models offering unique efficiency characteristics for specific deployment scenarios.

### State Space Models (SSM)

**Mamba series** (`state-spaces/mamba-*`) provides **linear-time inference** with no attention mechanism, ideal for very long sequences:
- `state-spaces/mamba-2.8b` — Best pure SSM model
- `state-spaces/mamba2-2.7b` — Improved SSD framework

**RWKV** (`RWKV/rwkv-6-world-7b`) offers **infinite context with constant memory** — no KV-cache required:
- Linear complexity regardless of sequence length
- Free text embeddings from hidden states
- 100% attention-free architecture

### Hybrid Architectures

**Jamba** (`ai21labs/AI21-Jamba-1.5-Large`) combines SSM + Transformer + MoE:
- 256K context window
- 2.5x faster inference than pure transformers
- First production-grade Mamba hybrid

**Nemotron-H** (`nvidia/Nemotron-H-56B-Base`) pioneered hybrid Mamba-2 + Transformer:
- ~8% attention layers, 92% Mamba-2
- 3x faster inference vs comparable transformers
- Up to 1M context in FP4

**Falcon-H1** (`tiiuae/Falcon-H1-7B-Instruct`) latest hybrid exploration:
- Transformer-SSM combination
- Optimized for efficiency

---

## Quick reference by license type

### Fully Permissive (Apache 2.0 / MIT)

Best for commercial deployment without restrictions:

- **Qwen2.5** (most sizes): `Qwen/Qwen2.5-*` (except 3B, 72B)
- **Mistral/Mixtral**: `mistralai/Mistral-*`, `mistralai/Mixtral-*`
- **Microsoft Phi**: `microsoft/phi-*`
- **OLMo**: `allenai/OLMo-*`
- **StarCoder2**: `bigcode/starcoder2-*`
- **DeepSeek-V3/R1**: `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-R1`
- **Falcon** (most): `tiiuae/falcon-*`
- **Mamba/RWKV**: `state-spaces/mamba-*`, `RWKV/*`
- **Granite**: `ibm-granite/granite-*`
- **Jamba v0.1**: `ai21labs/Jamba-v0.1`

### Community Licenses (Commercial OK with Attribution)

- **Llama 3.x**: `meta-llama/*` — 700M MAU threshold
- **Gemma**: `google/gemma-*` — Gemma Terms of Use
- **Yi**: `01-ai/Yi-*` — Yi License

### Research/Non-Commercial

- **Codestral**: `mistralai/Codestral-22B-v0.1` — MNPL-0.1
- **Command R/R+**: `CohereForAI/c4ai-command-r-*` — CC-BY-NC

---

## Deployment size estimates

For planning infrastructure, approximate VRAM requirements:

| Precision | Formula | Example (70B) |
|-----------|---------|---------------|
| FP32 | params × 4 bytes | ~280 GB |
| BF16/FP16 | params × 2 bytes | ~140 GB |
| FP8 | params × 1 byte | ~70 GB |
| INT4/GPTQ | params × 0.5 bytes | ~35 GB |

**MoE models** require full parameter storage but only activate a subset during inference — plan storage for total params, compute for active params.