"""Copy the files the Docker image needs into a staging folder and upload them to a Hugging Face Space."""
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["Dockerfile", ".dockerignore", "requirements-server.txt"]
DIRS = ["app", "frontend"]
SPACE_README = """---
title: ragGeneral
emoji: 🌳
colorFrom: green
colorTo: green
sdk: docker
app_port: 8080
pinned: false
---
Chat with the PDFs in `data/`. Source: GitHub repo ragGeneral (deployed automatically from main).
"""


def stage(dest: Path):
    for f in FILES:
        shutil.copy2(ROOT / f, dest / f)
    for d in DIRS:
        shutil.copytree(ROOT / d, dest / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for pdf in (ROOT / "data").rglob("*.pdf"):
        target = dest / pdf.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, target)
    (dest / "README.md").write_text(SPACE_README, encoding="utf-8")


def main():
    from huggingface_hub import HfApi

    space, token = os.environ["HF_SPACE"], os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    if os.getenv("GROQ_API_KEY"):
        api.add_space_secret(space, "GROQ_API_KEY", os.environ["GROQ_API_KEY"])
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space",
                          commit_message=f"Deploy {os.getenv('GITHUB_SHA', 'local')[:7]}",
                          delete_patterns=["app/**", "frontend/**", "data/**"])   # mirror removed files too
    print(f"Uploaded. The Space builds now: https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
