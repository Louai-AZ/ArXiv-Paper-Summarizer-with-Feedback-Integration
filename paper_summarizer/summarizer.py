from paper_summarizer.config import *
from paper_summarizer.services.langchain_service import get_prompt_with_fewshots
from paper_summarizer.services.huggingface_service import get_llm
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
import uuid

client = Client()

def summarizer_pipeline(messages, temperature=1.0):
    run_id = uuid.uuid4()
    llm = get_llm(temperature)
    prompt = get_prompt_with_fewshots()
    summarizer = (prompt | llm | StrOutputParser()).with_config(run_name="Summarizer")
    write_stream = summarizer.stream({"messages": [tuple(msg[:2]) for msg in messages]}, config={"run_id": run_id})

    full_response = ""
    for chunk in write_stream:
        full_response += chunk
    presigned = client.create_presigned_feedback_token(run_id, feedback_key="summary_quality")
    return full_response, presigned.url
