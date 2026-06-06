#!/usr/bin/env python3
import os
import sys

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "atx/gemma4-12b-mtp-assistant"


def main() -> None:
    repo_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=".",
        commit_message="Add validated Gemma 4 12B MTP assistant GGUFs",
    )
    print(f"uploaded https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
