import functools
from typing import Optional, cast
import uuid
import streamlit as st
from streamlit_feedback import streamlit_feedback
from langsmith import Client
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
import random
import re
from langchain import hub
from langchainhub import Client as HubClient
import logging
import concurrent.futures
from dotenv import load_dotenv
import os
from huggingface_hub import login


# Load environment variables
login(os.getenv('HUGGINGFACEHUB_API_TOKEN'))
load_dotenv()

# Setup logging for debugging and tracking
logging.basicConfig(level=logging.INFO)
st.set_page_config(
    page_title="Paper Summarizer with Feedback",  # Set page title for Streamlit app
    page_icon="📄🧠",  # Set app icon
)

logger = logging.getLogger(__name__)

# Define dataset and prompt names
DATASET_NAME = "Paper Summarizer"
PROMPT_NAME = "louup/tweet-critic-fewshot"
OPTIMIZER_PROMPT_NAME = "louup/convo-optimizer"
NUM_FEWSHOTS = 10
PROMPT_UPDATE_BATCHSIZE = 5

# Sidebar UI setup
st.sidebar.title("Session Information")
prompt_version = st.sidebar.text_input("Prompt Version", value="latest")
prompt_url = f"https://smith.langchain.com/hub/{PROMPT_NAME}"
if prompt_version and prompt_version != "latest":
    prompt_url = f"{prompt_url}/{prompt_version}"
st.sidebar.markdown(f"[See Prompt in Hub]({prompt_url})")
optimizer_prompt_url = f"https://smith.langchain.com/hub/{OPTIMIZER_PROMPT_NAME}"
st.sidebar.markdown(f"[See Optimizer Prompt in Hub]({optimizer_prompt_url})")

# Set up the LLMs for summarization and optimizer
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 1.0, 0.1)
chat_llm = HuggingFaceEndpoint(repo_id=repo_id, task="text-generation", temperature=temperature)
optimizer_llm = HuggingFaceEndpoint(repo_id=repo_id, task="text-generation", temperature=temperature)

# Initialize LangChain client to interact with dataset and example management
client = Client()

# Format examples for displaying in few-shot learning
def _format_example(example):
    return f"""<example>
    <original>
    {example.inputs['input']}
    </original>
    <summary>
    {example.outputs['output']}
    </summary>
</example>"""

# Fetch few-shot examples for training from dataset
def few_shot_examples():
    if client.has_dataset(dataset_name=DATASET_NAME):
        examples = list(client.list_examples(dataset_name=DATASET_NAME))
        if not examples:
            return ""
        examples = random.sample(examples, min(len(examples), NUM_FEWSHOTS))
        return "\n".join([_format_example(e) for e in examples])
    return ""

# Store the few-shot examples in session state for reuse
if st.session_state.get("few_shots"):
    few_shots = st.session_state.get("few_shots")
else:
    few_shots = few_shot_examples()
    st.session_state["few_shots"] = few_shots

# Pull prompt template from LangChain Hub and format it with few-shot examples
prompt = hub.pull(PROMPT_NAME + (f":{prompt_version}" if prompt_version and prompt_version != "latest" else ""))
prompt = prompt.partial(examples=few_shots)
summarizer = (prompt | chat_llm | StrOutputParser()).with_config(run_name="Summarizer")

# Fetch the full text of a paper using arXiv ID
def fetch_arxiv_full_text(paper_id: str) -> str:
    from langchain_community.document_loaders import ArxivLoader
    loader = ArxivLoader(query=paper_id, load_max_docs=1)
    docs = loader.load()
    if not docs:
        raise ValueError("No paper found with that ID")
    return docs[0].page_content

# Parse the summary and allow editing for feedback
def parse_summary(response: str, turn: int, box=None):
    box = box or st
    match = re.search(r"(.*?)<summary>(.*?)</summary>(.*?)", response.strip(), re.DOTALL)

    if match:
        pre, summary, post = match.groups()
        if pre:
            box.markdown(pre)
        box.markdown("✏️ **Modify if needed**: If the suggestion is close but needs tweaks, edit the summary directly in the text box before clicking 👍.")
        summary = st.text_area(
            "Edit this summary:",
            summary.strip(),
            key=f"summary_{turn}",
            height=400,
        )
        if post:
            box.markdown(post)
    else:
        summary = st.text_area(
            "Model response (editable fallback):",
            response.strip(),
            key=f"summary_fallback_{turn}",
            height=400,
        )
    return summary


# Log feedback from the user and update the model
def log_feedback(
    value: dict,
    *args,
    presigned_url: str,
    original_input: Optional[str] = None,
    txt: Optional[str] = None,
    **kwargs,
):
    # Mark session as ended and thank the user
    st.session_state["session_ended"] = True
    st.write("Thank you for your feedback! Updating summarizer...")

    # Use multithreading to handle feedback and model updates asynchronously
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        score = {"👍": 1, "👎": 0}.get(value["score"]) or 0  # Map thumbs to score
        comment = value.get("text")  # Get comment text
        futures.append(
            executor.submit(
                client.create_feedback_from_token,
                presigned_url,
                score=int(score),
                comment=comment,
            )
        )

        # Create a new example for training the model if feedback is positive
        def create_example():
            try:
                client.create_example(
                    inputs={"input": original_input},
                    outputs={"output": txt},
                    dataset_name=DATASET_NAME,
                )
                st.write("Example saved.")
            except:
                client.create_dataset(dataset_name=DATASET_NAME)
                client.create_example(
                    inputs={"input": original_input},
                    outputs={"output": txt},
                    dataset_name=DATASET_NAME,
                )
                st.write("Example saved.")

        if score and original_input and txt:
            futures.append(executor.submit(create_example))

        # Update the summarizer's instructions based on feedback
        with st.spinner("Updating instructions..."):
            def parse_updated_prompt(system_prompt_txt):
                return system_prompt_txt.split("<improved_prompt>")[1].split("</improved_prompt>")[0].strip()

            def format_conversation(messages):
                return "\n".join([f"<turn idx={i}>\n{msg[0]}: {msg[1]}\n</turn idx={i}>" for i, msg in enumerate(messages)])

            # Fetch the latest prompt updates from LangChain Hub
            hub_client = HubClient()
            hashes = [c["commit_hash"] for c in hub_client.list_commits(PROMPT_NAME)["commits"][:PROMPT_UPDATE_BATCHSIZE]]
            updated_prompts = [hub.pull(f"{PROMPT_NAME}:{h}") for h in hashes]
            optimizer_prompt = hub.pull(OPTIMIZER_PROMPT_NAME)

            # Optimize the prompt based on conversation history
            optimizer = (
                optimizer_prompt | optimizer_llm | StrOutputParser() | parse_updated_prompt
            ).with_config(run_name="Optimizer")

            # Format the conversation history for prompt update
            conversation = format_conversation(st.session_state.get("langchain_messages", []))
            if score:
                conversation = f'<rating>User rated this {value["score"]}</rating>\n\n' + conversation
            updated_sys_prompt = optimizer.invoke({
                "prompt_versions": "\n\n".join([f"<prompt version={h}>\n{cast(SystemMessagePromptTemplate, p.messages[0]).prompt.template}\n</prompt>" for h, p in zip(hashes, updated_prompts)]),
                "current_prompt": cast(SystemMessagePromptTemplate, prompt.messages[0]).prompt.template,
                "conversation": conversation,
                "final_value": txt,
            })
            updated_prompt = ChatPromptTemplate.from_messages([("system", updated_sys_prompt), MessagesPlaceholder(variable_name="messages")])
            hub.push(PROMPT_NAME, updated_prompt)
            st.success("Summarizer updated!")
        concurrent.futures.wait(futures)


# Display conversation messages and handle user feedback
messages = st.session_state.get("langchain_messages", [])
original_input = messages[0][1] if messages else None
for i, msg in enumerate(messages):
    with st.chat_message(msg[0]):
        if i == len(messages) - 1 and len(msg) == 3:
            updated = parse_summary(msg[1], i)
            presigned_url = msg[2]
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                on_submit=functools.partial(
                    log_feedback,
                    presigned_url=presigned_url,
                    original_input=original_input,
                    txt=updated,  
                ),
                key=f"fb_{i}",
            )
        else:
            updated = None
            st.markdown(msg[1])
            presigned_url = None

# Generate a unique ID for the run and prepare for feedback
run_id = uuid.uuid4()
presigned = client.create_presigned_feedback_token(run_id, feedback_key="summary_quality")

# Handle session end and reset
if st.session_state.get("session_ended"):
    st.write("Thanks for the feedback! This session has ended.")
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()
else:
    if not messages:
        st.write("Paste an arXiv ID (e.g., 2404.12345) or link to summarize:")
    if user_input := st.chat_input(placeholder="Paste arXiv ID or link here..."):
        st.chat_message("user").write(user_input)
        paper_id_match = re.search(r"(\d{4}\.\d{5})", user_input)
        if not paper_id_match:
            st.error("Couldn't extract paper ID from input. Try a valid arXiv ID.")
        else:
            paper_id = paper_id_match.group(1)
            content = fetch_arxiv_full_text(paper_id)
            messages.append(("user", content))
            with st.chat_message("assistant"):
                write_stream = summarizer.stream({"messages": [tuple(msg[:2]) for msg in messages]}, config={"run_id": run_id})
                message_placeholder = st.empty()
                full_response = ""
                for chunk in write_stream:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown("")
                summary_txt = parse_summary(full_response, len(messages), message_placeholder)
                messages.append(("assistant", full_response, presigned.url))
            st.session_state["langchain_messages"] = messages
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                on_submit=functools.partial(
                    log_feedback,
                    presigned_url=presigned.url,
                    original_input=content,
                    txt=summary_txt,
                ),
                key=f"fb_{len(messages) - 1}",
            )
