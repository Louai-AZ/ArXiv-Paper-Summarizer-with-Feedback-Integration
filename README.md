# ArXiv-Paper-Summarizer-with-Feedback-Integration

## Project Overview

This project is a web-based application that allows users to input an arXiv paper ID (or link) and get a summary generated using a language model. The summarization process is interactive, with the ability for users to provide feedback on the summary. This feedback helps improve the summarizer over time.

The system is built using **Streamlit**, **LangChain**, **Langsmith** and **HuggingFace** models, with the ability to pull papers from **arXiv** and dynamically summarize them based on user input.

---


## Key Features

- 📥 Fetch papers from **arXiv** using paper ID or link.
- 🧠 Summarize the content..
- 👍👎 Let users rate the summary and save feedback for model improvement.
- 💬 Supports prompt updating and feedback-based optimization.

---
 
## Project Structure

```bash
├── appV2.py                 # The whole application code (you can run this directly: streamlit run appV2.py)

├── main.py                 # Entry point for the Streamlit app
paper_summarizer/
├── config.py               # Global constants and configuration
├── arxiv_fetcher.py        # Logic to fetch arXiv paper content
├── feedback.py             # Feedback handling and LangSmith interaction
├── summarizer.py           # Summary generation logic
├── utils.py                # Helper functions
├── services/
│   ├── langchain_service.py     # Langchain summarizer setup
│   └── huggingface_service.py  # HuggingFace model setup
├── .env                    # Your secret API keys
├── requirements.txt        # Python dependencies
└── README.md               # This file!
```

---

## Setup Instructions

1. **Clone the repository**
``` bash 
git clone https://github.com/Louai-AZ/ArXiv-Paper-Summarizer-with-Feedback-Integration.git
```

2. **Create a virtual environment**
``` bash 
python -m venv .xtwit
.\\.xtwit\\Scripts\\Activate.ps1  # On Windows PowerShell
```

3. **Install dependencies**
``` bash 
pip install -r requirements.txt
```

4. **Add environment variables in .env file**
``` bash 
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=your_project_name
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
```

5. **Run the app**
``` bash 
streamlit run main.py
```

6. **Open your browser**
``` bash 
Go to: http://localhost:8501
```

---


## How to Use

1. **Enter the Paper ID**  
   On the main page, paste an arXiv ID (e.g., `2404.12345`) or a full arXiv paper URL into the input box.

2. **View the Summary**  
   After entering the paper ID or link, the system will fetch the paper and generate a summary.

3. **Provide Feedback**
   - 👍 **Like the suggestion**: Click the thumbs-up button to save the modified summary as an approved example.
   - 👎 **Dislike the suggestion**: Click the thumbs-down button.
   - ✏️ **Modify if needed**: If the suggestion is close but needs tweaks, edit the summary directly in the text box before clicking 👍.

   > ⚠️ **Note**: Once you click either **👍** or **👎**, the conversation ends, and the bot will no longer respond in that session.

4. **Reset the Session**  
   To start a new conversation, click the "Reset" button or refresh your browser.

5. **Learning from Feedback**  
   The application will learn from your feedback and adapt future suggestions accordingly.
