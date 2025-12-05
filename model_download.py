from huggingface_hub import snapshot_download, login
import os
from datetime import datetime

OUTPUT_BASE = OUTPUT_BASE = "/Volumes/robots/models"

MODELS = [
  {
    "id": "Qwen/QwQ-32B",
    "name": "QwQ-32B",
    "size_gb": 65,
    "category": "reasoning",
    "description": "o1-like deep reasoning, best open reasoning model"
  },
  {
    "id": "Qwen/Qwen2.5-72B-Instruct",
    "name": "Qwen2.5-72B-Instruct",
    "size_gb": 144,
    "category": "general",
    "description": "Best open 70B+ model, 18T tokens, 128K context, Apache 2.0"
  },
  {
    "id": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "name": "Qwen2.5-Coder-32B",
    "size_gb": 64,
    "category": "coding",
    "description": "Matches GPT-4o on coding, 40+ languages"
  },
  {
    "id": "nvidia/OpenReasoning-Nemotron-32B",
    "name": "OpenReasoning-Nemotron-32B",
    "size_gb": 64,
    "category": "reasoning",
    "description": "SOTA math/code/science, distilled from R1"
  },
  {
    "id": "meta-llama/Llama-3.3-70B-Instruct",
    "name": "Llama-3.3-70B-Instruct",
    "size_gb": 140,
    "category": "general",
    "description": "405B performance at 70B cost, production-grade"
  },
  {
    "id": "microsoft/phi-4",
    "name": "Phi-4",
    "size_gb": 28,
    "category": "reasoning",
    "description": "80.4% MATH, punches above weight at 14B"
  },
  {
    "id": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "name": "Nemotron-Nano-9B",
    "size_gb": 18,
    "category": "general",
    "description": "Hybrid Mamba-Transformer, 3-6x faster"
  },
  {
    "id": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "name": "DeepSeek-Coder-6.7B",
    "size_gb": 13,
    "category": "coding",
    "description": "Fast coding baseline, 86 languages"
  },
  {
    "id": "Qwen/Qwen2.5-Math-7B-Instruct",
    "name": "Qwen2.5-Math-7B",
    "size_gb": 14,
    "category": "math",
    "description": "Math reasoning specialist"
  },
  {
    "id": "deepseek-ai/DeepSeek-V3",
    "name": "DeepSeek-V3",
    "size_gb": 685,
    "category": "frontier",
    "description": "Matches GPT-4o, MIT license"
  }
]

def print_collection_info():
  total_size = sum(m["size_gb"] for m in MODELS)
  
  print("\n" + "="*80)
  print("CURATED REASONING & CODING COLLECTION")
  print("="*80)
  print(f"Total models: {len(MODELS)}")
  print(f"Estimated size: ~{total_size} GB")
  print(f"Output directory: {OUTPUT_BASE}")
  print("\n")
  
  categories = {}
  for m in MODELS:
    cat = m["category"]
    if cat not in categories:
      categories[cat] = []
    categories[cat].append(m)
  
  for cat, models in sorted(categories.items()):
    cat_size = sum(m["size_gb"] for m in models)
    print(f"{cat.upper()} ({cat_size} GB)")
    for m in models:
      print(f"   {m['name']:30} {m['size_gb']:3}GB - {m['description']}")
    print()
  
  print("="*80 + "\n")

def download_model(model_info, index, total):
  model_id = model_info["id"]
  local_dir = os.path.join(OUTPUT_BASE, model_info["name"])
  
  print("\n" + "="*80)
  print(f"[{index}/{total}] DOWNLOADING: {model_info['name']}")
  print("="*80)
  print(f"Repo ID:      {model_id}")
  print(f"Category:     {model_info['category']}")
  print(f"Est. Size:    ~{model_info['size_gb']} GB")
  print(f"Description:  {model_info['description']}")
  print(f"Save Path:    {local_dir}")
  print(f"Started:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  print("="*80 + "\n")
  
  try:
    snapshot_download(
      repo_id=model_id,
      local_dir=local_dir,
      local_dir_use_symlinks=False,
      resume_download=True,
      max_workers=4,
      allow_patterns=[
        "*.safetensors",
        "*.json",
        "*.txt",
        "*.md",
        "*.model",
        "tokenizer*",
        "LICENSE*",
        "*.tiktoken"
      ],
      ignore_patterns=[
        "*.bin",
        "*.pt",
        "*.pth",
        "*.msgpack",
        "*.h5",
        "*.ckpt"
      ]
    )
    
    # Clean up cache immediately after successful download
    cache_dir = os.path.join(local_dir, ".cache")
    if os.path.exists(cache_dir):
      import shutil
      shutil.rmtree(cache_dir)
      print(f"Cleaned up cache for {model_info['name']}")
      
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\nSUCCESS: {model_info['name']} - Completed at {end_time}\n")
    return True
      
  except Exception as e:
      print(f"\nERROR: {model_info['name']}")
      print(f"Error message: {str(e)}\n")
      return False

def select_models():
  print("Select models to download:")
  print("0. Download ALL models")
  
  for i, model in enumerate(MODELS, 1):
    print(f"{i}. {model['name']} ({model['size_gb']}GB) - {model['description']}")
  
  print(f"\nEnter numbers separated by commas (e.g., 1,3,5)")
  print(f"Or enter 0 to download all, or 'q' to quit")
  
  choice = input("\nYour selection: ").strip()
  
  if choice.lower() == 'q':
    return None
  
  if choice == '0':
    return MODELS
  
  try:
    indices = [int(x.strip()) for x in choice.split(',')]
    selected = [MODELS[i-1] for i in indices if 1 <= i <= len(MODELS)]
    
    if not selected:
      print("No valid models selected")
      return None
    
    total_size = sum(m["size_gb"] for m in selected)
    print(f"\nSelected {len(selected)} models (~{total_size} GB)")
    return selected
    
  except (ValueError, IndexError):
    print("Invalid selection")
    return None

def main():
  print_collection_info()
  
  selected_models = select_models()
  
  if selected_models is None:
    print("Download cancelled.")
    return
  
  total_size = sum(m["size_gb"] for m in selected_models)
  print(f"\nAbout to download {len(selected_models)} models (~{total_size} GB)")
  confirm = input("Proceed? (y/n): ").strip().lower()
  
  if confirm != 'y':
    print("Download cancelled.")
    return
  
  os.makedirs(OUTPUT_BASE, exist_ok=True)
  
  print("\n" + "="*80)
  print("STARTING DOWNLOADS")
  print("="*80)
  
  results = []
  total = len(selected_models)
  
  for i, model in enumerate(selected_models, 1):
    success = download_model(model, i, total)
    results.append((model["name"], model["size_gb"], success))
  
  print("\n" + "="*80)
  print("DOWNLOAD SUMMARY")
  print("="*80)
  
  successful_count = 0
  failed_count = 0
  total_downloaded_gb = 0
  
  for name, size_gb, success in results:
    if success:
      status = "SUCCESS"
      successful_count += 1
      total_downloaded_gb += size_gb
    else:
      status = "FAILED"
      failed_count += 1
    
    print(f"{status:12} {name:30} ({size_gb} GB)")
  
  print("\n" + "-"*80)
  print(f"Successful: {successful_count}/{total} models")
  print(f"Failed:     {failed_count}/{total} models")
  print(f"Downloaded: ~{total_downloaded_gb} GB")
  print(f"Completed:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  print("="*80 + "\n")

if __name__ == "__main__":
  main()