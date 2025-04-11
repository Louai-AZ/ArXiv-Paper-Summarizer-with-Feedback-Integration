import streamlit as st
from paper_summarizer.arxiv_fetcher import fetch_arxiv_full_text
from paper_summarizer.summarizer import summarizer_pipeline
from paper_summarizer.feedback import handle_feedback, display_feedback_ui

st.set_page_config(page_title="Paper Summarizer", page_icon="📄🧠")
st.sidebar.title("Session Info")
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 1.0, 0.1)

if st.session_state.get("session_ended"):
    st.write("Thanks for the feedback! This session has ended.")
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()
else:
    messages = st.session_state.get("langchain_messages", [])
    if not messages:
        st.write("Paste an arXiv ID (e.g., 2404.12345) or link to summarize:")

    if user_input := st.chat_input("Paste arXiv ID or link here..."):
        st.chat_message("user").write(user_input)
        paper_id = fetch_arxiv_full_text(user_input)
        if paper_id:
            content = paper_id["content"]
            messages.append(("user", content))
            st.session_state["langchain_messages"] = messages

            summary, presigned_url = summarizer_pipeline(messages, temperature)
            messages.append(("assistant", summary, presigned_url))
            st.session_state["langchain_messages"] = messages

            display_feedback_ui(messages, summary, presigned_url, content)
        else:
            st.error("Could not fetch paper from arXiv ID.")
    else:
        for i, msg in enumerate(messages):
            with st.chat_message(msg[0]):
                st.markdown(msg[1])
