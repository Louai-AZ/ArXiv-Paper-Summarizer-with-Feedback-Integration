import os
from dotenv import load_dotenv

load_dotenv()

DATASET_NAME = "Paper Summarizer"
PROMPT_NAME = "louup/tweet-critic-fewshot"
OPTIMIZER_PROMPT_NAME = "louup/convo-optimizer"
NUM_FEWSHOTS = 10
PROMPT_UPDATE_BATCHSIZE = 5
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
