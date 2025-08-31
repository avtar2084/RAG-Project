# RAG-Project
# RAG-Project

## Overview

RAG-Project is a collection of tools and modules for summarizing YouTube videos using Retrieval-Augmented Generation (RAG) techniques. It combines transcript extraction, semantic search, and large language models to provide accurate summaries and answers to user queries.

## Features

- Fetches and processes YouTube video transcripts
- Chunks transcripts for efficient retrieval
- Uses FAISS for similarity search and context retrieval
- Integrates IBM WatsonX LLM for summarization and Q&A
- Interactive Gradio interface for user input
- Modular design for easy extension

## Folder Structure

- `YouTubeSummarizer/` — Main module for transcript extraction, summarization, and Q&A
- `recommendation_system/` — (If present) Tools for recommending related content
- `README.md` — Project documentation

## Getting Started

1. Clone the repository:
   ```
   git clone <repo-url>
   cd RAG-Project
   ```
2. Install dependencies for each module:
   ```
   pip install -r YouTubeSummarizer/requirements.txt
   ```
3. Run the YouTubeSummarizer:
   ```
   python YouTubeSummarizer/ytbot.py
   ```
4. Use the Gradio interface to summarize videos or ask questions.

## Requirements

- Python 3.8+
- IBM WatsonX credentials (for LLM and embeddings)
- Internet connection (for fetching YouTube transcripts)

## Example Workflow

1. Enter a YouTube video URL in the interface.
2. The system fetches and processes the transcript.
3. Summarize the video or ask questions about its content.
4. Relevant context is retrieved using FAISS and answers are generated with WatsonX LLM.

## License

MIT License

## Acknowledgements

- IBM WatsonX
- FAISS
- Gradio
-