#!/usr/bin/env python3
import argparse
import os
import shutil
from datetime import datetime
from huggingface_hub import snapshot_download, login
from model_lists import dec_2025

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass 

# Configuration from environment
OUTPUT_BASE = os.environ.get("OUTPUT_BASE")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Authenticate with Hugging Face
if HF_TOKEN:
    login(token=HF_TOKEN)

def print_collection_info(models):
    """Print summary information about the model collection."""
    total_size = sum(m["size_gb"] for m in models)

    print("\n" + "="*80)
    print("MODEL COLLECTION")
    print("="*80)
    print(f"Total models: {len(models)}")
    print(f"Estimated size: ~{total_size} GB")
    print(f"Output directory: {OUTPUT_BASE}")
    print("\n")

    # Group by category
    categories = {}
    for m in models:
        cat = m["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(m)

    for cat, cat_models in sorted(categories.items()):
        cat_size = sum(m["size_gb"] for m in cat_models)
        print(f"{cat.upper()} ({cat_size} GB)")
        for m in cat_models:
            print(f"   {m['name']:30} {m['size_gb']:3}GB - {m['description']}")
        print()

    print("="*80 + "\n")


def download_model(model_info, index, total):
    """Download a single model from Hugging Face."""
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

        # Clean up cache
        cache_dir = os.path.join(local_dir, ".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleaned up cache for {model_info['name']}")

        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n✓ SUCCESS: {model_info['name']} - Completed at {end_time}\n")
        return True

    except Exception as e:
        print(f"\n✗ ERROR: {model_info['name']}")
        print(f"Error message: {str(e)}\n")
        return False


def select_models_interactive(available_models):
    """Interactive model selection interface."""
    print("Select models to download:")
    print("0. Download ALL models")

    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model['name']} ({model['size_gb']}GB) - {model['description']}")

    print("\nEnter numbers separated by commas (e.g., 1,3,5)")
    print("Or enter 0 to download all, or 'q' to quit")

    choice = input("\nYour selection: ").strip()

    if choice.lower() == 'q':
        return None

    if choice == '0':
        return available_models

    try:
        indices = [int(x.strip()) for x in choice.split(',')]
        selected = [available_models[i-1] for i in indices if 1 <= i <= len(available_models)]

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
    parser = argparse.ArgumentParser(
        description="Download LLM models from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (select from all models)
  python3 droid_factory.py

  # Download specific models by name
  python3 droid_factory.py --models "Qwen2.5-72B-Instruct" "Phi-4"

  # Download all models in a category
  python3 droid_factory.py --category coding

  # Download all small models
  python3 droid_factory.py --size small

  # List available categories
  python3 droid_factory.py --list-categories

  # List available size ranges
  python3 droid_factory.py --list-sizes

Environment Variables:
  HF_TOKEN      Your Hugging Face API token (required for gated models)
  OUTPUT_BASE   Download directory (default: /Volumes/robots/models)
        """
    )

    parser.add_argument(
        '--models',
        nargs='+',
        help='Specific model names to download'
    )
    parser.add_argument(
        '--category',
        help='Download all models in a category (e.g., coding, math, reasoning)'
    )
    parser.add_argument(
        '--size',
        help='Download models by size (small, medium, large, xlarge, massive)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available models'
    )
    parser.add_argument(
        '--list-categories',
        action='store_true',
        help='List available categories and exit'
    )
    parser.add_argument(
        '--list-sizes',
        action='store_true',
        help='List size categories and exit'
    )
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Skip confirmation prompts'
    )

    args = parser.parse_args()

    # Handle list commands
    if args.list_categories:
        print("\nAvailable Categories:")
        for cat, name in sorted(dec_2025.CATEGORIES.items()):
            models = dec_2025.get_models_by_category(cat)
            total_size = sum(m["size_gb"] for m in models)
            print(f"  {cat:15} - {name:25} ({len(models):3} models, ~{total_size:6.0f} GB)")
        return

    if args.list_sizes:
        print("\nSize Categories:")
        for size, name in sorted(dec_2025.SIZE_CATEGORIES.items()):
            models = dec_2025.get_models_by_size(size)
            total_size = sum(m["size_gb"] for m in models)
            print(f"  {size:10} - {name:25} ({len(models):3} models, ~{total_size:6.0f} GB)")
        return

    # Determine which models to download
    selected_models = None

    if args.all:
        selected_models = dec_2025.get_all_models()
        print(f"Selected: ALL models ({len(selected_models)} total)")

    elif args.category:
        selected_models = dec_2025.get_models_by_category(args.category)
        if not selected_models:
            print(f"Error: No models found for category '{args.category}'")
            print("Run with --list-categories to see available options")
            return
        print(f"Selected: Category '{args.category}' ({len(selected_models)} models)")

    elif args.size:
        selected_models = dec_2025.get_models_by_size(args.size)
        if not selected_models:
            print(f"Error: No models found for size '{args.size}'")
            print("Run with --list-sizes to see available options")
            return
        print(f"Selected: Size category '{args.size}' ({len(selected_models)} models)")

    elif args.models:
        # Find models by name
        all_models = dec_2025.get_all_models()
        selected_models = [m for m in all_models if m["name"] in args.models]

        if len(selected_models) != len(args.models):
            found = {m["name"] for m in selected_models}
            missing = set(args.models) - found
            print(f"Warning: Could not find models: {missing}")

        if not selected_models:
            print("Error: No models found with those names")
            return

        print(f"Selected: {len(selected_models)} specific models")

    else:
        # Interactive mode - show all models
        all_models = dec_2025.get_all_models()
        print(f"\nInteractive Mode - {len(all_models)} models available")
        print_collection_info(all_models)
        selected_models = select_models_interactive(all_models)

    if not selected_models:
        print("No models selected. Exiting.")
        return

    # Show selection
    print_collection_info(selected_models)

    # Confirm download
    if not args.non_interactive:
        total_size = sum(m["size_gb"] for m in selected_models)
        print(f"\nAbout to download {len(selected_models)} models (~{total_size} GB)")
        print(f"Download location: {OUTPUT_BASE}")
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Download cancelled.")
            return

    # Create output directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Download models
    print("\n" + "="*80)
    print("STARTING DOWNLOADS")
    print("="*80)

    results = []
    total = len(selected_models)

    for i, model in enumerate(selected_models, 1):
        success = download_model(model, i, total)
        results.append((model["name"], model["size_gb"], success))

    # Print summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)

    successful = sum(1 for _, _, success in results if success)
    failed = sum(1 for _, _, success in results if not success)
    total_downloaded_gb = sum(size for _, size, success in results if success)

    for name, size_gb, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status:12} {name:30} ({size_gb} GB)")

    print("\n" + "-"*80)
    print(f"Successful: {successful}/{total} models")
    print(f"Failed:     {failed}/{total} models")
    print(f"Downloaded: ~{total_downloaded_gb} GB")
    print(f"Completed:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
