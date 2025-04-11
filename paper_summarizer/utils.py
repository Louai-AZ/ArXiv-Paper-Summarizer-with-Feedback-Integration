import re
import streamlit as st

def parse_summary(response: str, turn: int, box=None):
    box = box or st
    match = re.search(r"(.*?)<summary>(.*?)</summary>(.*?)", response.strip(), re.DOTALL)
    if match:
        pre, summary, post = match.groups()
        if pre:
            box.markdown(pre)
        box.markdown("✏️ **Modify if needed**:")
        summary = st.text_area("Edit this summary:", summary.strip(), key=f"summary_{turn}", height=400)
        if post:
            box.markdown(post)
    else:
        summary = st.text_area("Model response:", response.strip(), key=f"summary_fallback_{turn}", height=400)
    return summary
