This function retrieves relevant context from the FAISS index based on the user's query and generates an answer using the preprocessed transcript. It first checks if the transcript has been fetched; if not, it fetches and processes the transcript from the provided YouTube video URL.

If the transcript is available and a user question is provided, the function proceeds to chunk the transcript for better context retrieval. It then sets up IBM Watson credentials and initializes the watsonx LLM specifically for Q&A tasks.

Next, it creates a FAISS index using the chunked transcript and sets up the Q&A prompt template and chain. Finally, it generates an answer to the user's question using the FAISS index and returns the answer. If the transcript hasn't been fetched or if the user fails to provide a valid question, the function returns a relevant message indicating the issue.

# YouTubeSummarizer

## Overview

YouTubeSummarizer is a tool that fetches transcripts from YouTube videos, summarizes their content, and answers user questions using IBM WatsonX LLM and FAISS for context retrieval.

## Features

- Extracts transcripts from YouTube videos ([`get_transcript`](YouTubeSummarizer/ytbot.py))
- Processes and chunks transcripts for efficient retrieval ([`process`](YouTubeSummarizer/ytbot.py), [`chunk_transcript`](YouTubeSummarizer/ytbot.py))
- Uses IBM WatsonX LLM for summarization and Q&A ([`initialize_watsonx_llm`](YouTubeSummarizer/ytbot.py))
- Builds FAISS index for similarity search ([`create_faiss_index`](YouTubeSummarizer/ytbot.py))
- Provides a Gradio interface for user interaction

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the bot:
   ```
   python ytbot.py
   ```
3. Use the Gradio interface to:
   - Enter a YouTube video URL
   - Summarize the video
   - Ask questions about the video content

## Main Components

- [`ytbot.py`](YouTubeSummarizer/ytbot.py): Main logic for transcript extraction, summarization, Q&A, and Gradio UI.
- [`requirements.txt`](YouTubeSummarizer/requirements.txt): Python dependencies.

## Example Workflow

1. Enter a YouTube URL.
2. Fetch and process the transcript.
3. Summarize the video or ask a question.
4. The system retrieves relevant context using FAISS and generates answers with WatsonX LLM.

## Notes

- Requires IBM WatsonX credentials for LLM and embeddings.
- Only English transcripts