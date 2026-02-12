#!/usr/bin/env python3
"""
Download and extract aloha-agilex_clean_50.zip files from RoboTwin2.0 dataset.
Usage: python download_aloha_dataset.py --output_dir /path/to/save
"""

import os
import zipfile
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm


REPO_ID = "TianxingChen/RoboTwin2.0"
TARGET_FILENAME = "aloha-agilex_clean_50.zip"


def get_all_tasks(repo_id):
    """Get all task names from the dataset repository."""
    print(f"Fetching file list from {repo_id}...")
    all_files = list_repo_files(repo_id, repo_type="dataset")
    
    # Extract unique task names from paths like "dataset/task_name/aloha-agilex_clean_50.zip"
    tasks = set()
    for file_path in all_files:
        if file_path.startswith("dataset/") and TARGET_FILENAME in file_path:
            # Extract task name from path
            parts = file_path.split("/")
            if len(parts) >= 3:
                task_name = parts[1]
                tasks.add(task_name)
    
    return sorted(list(tasks))


def download_and_extract(task_name, output_dir):
    """Download and extract zip file for a specific task."""
    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Path in the repo
    repo_file_path = f"dataset/{task_name}/{TARGET_FILENAME}"
    
    try:
        print(f"\n[{task_name}] Downloading {TARGET_FILENAME}...")
        
        # Download the file
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_file_path,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"[{task_name}] Downloaded to: {downloaded_path}")
        
        # Extract the zip file
        print(f"[{task_name}] Extracting...")
        with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
            zip_ref.extractall(task_output_dir)
        
        print(f"[{task_name}] Extracted to: {task_output_dir}")
        
        # Delete the zip file
        os.remove(downloaded_path)
        print(f"[{task_name}] Deleted zip file: {downloaded_path}")
        
        return True
        
    except Exception as e:
        print(f"[{task_name}] Error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract aloha-agilex_clean_50.zip files from RoboTwin2.0 dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save and extract files"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific tasks to download (default: all tasks)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get task list
    if args.tasks:
        tasks = args.tasks
        print(f"Downloading {len(tasks)} specified tasks...")
    else:
        tasks = get_all_tasks(REPO_ID)
        print(f"Found {len(tasks)} tasks with {TARGET_FILENAME}")
    
    print(f"Tasks: {', '.join(tasks)}\n")
    
    # Download and extract each task
    success_count = 0
    failed_tasks = []
    
    for i, task_name in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"Processing task {i}/{len(tasks)}: {task_name}")
        print(f"{'='*60}")
        
        if download_and_extract(task_name, output_dir):
            success_count += 1
        else:
            failed_tasks.append(task_name)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Download Summary")
    print(f"{'='*60}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_tasks)}")
    
    if failed_tasks:
        print(f"\nFailed tasks:")
        for task in failed_tasks:
            print(f"  - {task}")


if __name__ == "__main__":
    main()