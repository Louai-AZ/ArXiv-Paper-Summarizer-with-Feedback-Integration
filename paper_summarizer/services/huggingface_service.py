from huggingface_hub import login
from paper_summarizer.config import HUGGINGFACE_REPO_ID, HUGGINGFACEHUB_API_TOKEN
from langchain_huggingface import HuggingFaceEndpoint

def get_llm(temperature=1.0):
    login(HUGGINGFACEHUB_API_TOKEN)
    return HuggingFaceEndpoint(repo_id=HUGGINGFACE_REPO_ID, task="text-generation", temperature=temperature)
