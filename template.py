import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# List all files and directories to scaffold
list_of_files = [
    'main.py',
    'paper_summarizer/config.py',
    'paper_summarizer/arxiv_fetcher.py',
    'paper_summarizer/feedback.py',
    'paper_summarizer/summarizer.py',
    'paper_summarizer/utils.py',
    'paper_summarizer/services/langchain_service.py',
    'paper_summarizer/services/huggingface_service.py',
    '.env',
    '.env.example',
    'requirements.txt',
]

for filepath in list_of_files:
    path = Path(filepath)
    filedir = path.parent
    filename = path.name

    # Create directories if they don't exist
    if filedir != Path('.'):
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file {filename}")

    # Create empty file if missing or empty
    if not path.exists() or path.stat().st_size == 0:
        path.touch()
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
