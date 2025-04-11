import re
from langchain_community.document_loaders import ArxivLoader

def fetch_arxiv_full_text(input_text: str):
    paper_id_match = re.search(r"(\d{4}\.\d{5})", input_text)
    if not paper_id_match:
        return None
    paper_id = paper_id_match.group(1)
    loader = ArxivLoader(query=paper_id, load_max_docs=1)
    docs = loader.load()
    if docs:
        return {"id": paper_id, "content": docs[0].page_content}
    return None
