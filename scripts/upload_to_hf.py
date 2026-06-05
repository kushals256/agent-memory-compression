import os
import sys
from huggingface_hub import HfApi

def upload():
    print("======================================================================")
    print("              Upload Dataset to Hugging Face Hub")
    print("======================================================================")
    
    # 1. Ask for credentials
    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("Error: Username cannot be empty.")
        return
        
    token = input("Enter your Hugging Face Write Token (get it from hf.co/settings/tokens): ").strip()
    if not token:
        print("Error: Token cannot be empty.")
        return
        
    repo_id = f"{username}/agent-memory-benchmark"
    
    # Files to upload
    files_to_upload = {
        "data/conversation.json": "data/conversation.json",
        "eval/questions.json": "eval/questions.json",
    }
    
    # Create the temporary README content for Hugging Face Dataset Card
    readme_content = """---
license: mit
task_categories:
- conversational
- question-answering
tags:
- agent-memory
- evaluation
- rag
size_categories:
- n < 1K
---

# Agent Memory Compression & Evaluation Benchmark

This dataset is a controlled evaluation testbed designed to benchmark long-term memory architectures for conversational AI agents. It stress-tests how agents handle long conversations with complex fact dynamics.

## Dataset Structure

### 1. `conversation.json`
A 100-turn synthetic conversation (50 user, 50 assistant turns) containing embedded facts categorized under:
- **Simple Facts**: Baseline retrieval details.
- **Deep/Early Facts**: Facts mentioned at Turn 3 and queried at Turn 95+.
- **Contradictions (Temporal)**: Facts that are overridden later (e.g., Favorite color blue at Turn 11 -> red at Turn 75).
- **Implicit Facts**: Facts requiring basic logical inference (e.g., mentioning "brother and sister" -> 2 siblings).
- **Repeated Facts**: Evaluates how memory strategies handle duplicate context reinforcement vs. deduplication.

### 2. `questions.json`
20 targeted evaluation questions mapped to the fact taxonomy with fixed ground truth answers, graded on a strict `0.0` / `0.5` / `1.0` scale.

## Usage
You can load the files directly in Python:

```python
import json

with open("data/conversation.json", "r") as f:
    conversation = json.load(f)

with open("eval/questions.json", "r") as f:
    questions = json.load(f)
```

For the full benchmark repository and visualization code, visit the GitHub repository:
[github.com/kushals256/agent-memory-compression](https://github.com/kushals256/agent-memory-compression)
"""
    
    temp_readme_path = "temp_readme.md"
    with open(temp_readme_path, "w") as f:
        f.write(readme_content)
        
    files_to_upload[temp_readme_path] = "README.md"
    
    try:
        api = HfApi()
        print(f"\nCreating dataset repository: {repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
        
        print("\nUploading files...")
        for local_path, repo_path in files_to_upload.items():
            if not os.path.exists(local_path):
                print(f"Error: Local file {local_path} not found.")
                continue
            print(f"  Uploading {local_path} -> {repo_path}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )
            
        print(f"\nSuccess! Dataset is live at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
    finally:
        # Clean up temp readme file
        if os.path.exists(temp_readme_path):
            os.remove(temp_readme_path)

if __name__ == "__main__":
    upload()
