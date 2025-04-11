import functools
from streamlit_feedback import streamlit_feedback
from paper_summarizer.utils import parse_summary
from paper_summarizer.config import DATASET_NAME
from langsmith import Client
import streamlit as st


client = Client()

def handle_feedback(value, presigned_url, original_input, txt):
    from paper_summarizer.services.langchain_service import update_prompt_from_feedback

    st.session_state["session_ended"] = True
    st.write("Thank you for your feedback! Updating summarizer...")

    score = {"👍": 1, "👎": 0}.get(value["score"]) or 0
    comment = value.get("text")

    client.create_feedback_from_token(presigned_url, score=int(score), comment=comment)
    if score and original_input and txt:
        try:
            client.create_example(
                inputs={"input": original_input},
                outputs={"output": txt},
                dataset_name=DATASET_NAME,
            )
        except:
            client.create_dataset(dataset_name=DATASET_NAME)
            client.create_example(
                inputs={"input": original_input},
                outputs={"output": txt},
                dataset_name=DATASET_NAME,
            )
    if score:
        update_prompt_from_feedback(value["score"], txt)

def display_feedback_ui(messages, summary_txt, presigned_url, original_input):
    turn_index = len(messages) - 1
    updated_summary = parse_summary(summary_txt, turn_index)
    streamlit_feedback(
        feedback_type="thumbs",
        on_submit=functools.partial(
            handle_feedback,
            presigned_url=presigned_url,
            original_input=original_input,
            txt=updated_summary,
        ),
        key=f"fb_{turn_index}",
    )
