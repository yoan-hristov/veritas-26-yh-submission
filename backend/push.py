"""Push backend files to a Hugging Face Space.

Setup:  pip install huggingface_hub
Token:  create a WRITE token at https://huggingface.co/settings/tokens
        then set it:  $env:HF_TOKEN="hf_xxx"   (PowerShell)
Usage:  python push.py
"""
import os
from pathlib import Path
from huggingface_hub import HfApi
from dotenv import load_dotenv
load_dotenv()

SPACE_ID = "YoanHristov/xraynpcbackend"  # e.g. YoanHristov/veritas
FILES = ["app.py", "README.md", "requirements.txt"]

token = os.getenv("HF_TOKEN")
if not token:
    raise SystemExit("Set HF_TOKEN first (write-access token).")

api = HfApi(token=token)
here = Path(__file__).parent
for name in FILES:
    api.upload_file(
        path_or_fileobj=str(here / name),
        path_in_repo=name,
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message=f"Update {name}",
    )
    print(f"Uploaded {name}")
print(f"Done: https://huggingface.co/spaces/{SPACE_ID}")
