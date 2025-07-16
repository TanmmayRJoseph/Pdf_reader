# RAG Agent: PDF Question Answering with Gemini & Chroma

This project is a Retrieval-Augmented Generation (RAG) agent that answers user questions based on the content of PDF documents. It uses Google Gemini for language understanding and Chroma for vector storage and retrieval.

## Features

- Loads and parses all PDFs in the `data/` directory
- Splits documents into overlapping text chunks for better retrieval
- Embeds chunks using Google Generative AI Embeddings
- Stores and retrieves document vectors with Chroma
- Uses Gemini (via LangChain) to synthesize answers from retrieved content
- Simple conversational flow managed by LangGraph

## Project Structure

```
.env
.gitignore
main.py
prototype.ipynb
.vscode/
    settings.json
data/
    sample1.pdf
    sample2.pdf
    sample3.pdf
```

## Setup

1. **Clone the repository and install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

2. **Add your Google API key:**
    - Create a `.env` file in the project root:
      ```
      GOOGLE_API_KEY=your-google-api-key-here
      ```

3. **Add PDF files:**
    - Place your PDF files in the `data/` directory.

## Usage

### Run as a Script

```sh
python main.py
```

- The script will load PDFs, build the vector database, and answer a sample question.
- You can modify the initial question in `main.py` or interactively prompt for questions.

### Jupyter Notebook

- See [`prototype.ipynb`](prototype.ipynb) for an interactive, step-by-step version.

## Requirements

- Python 3.8+
- [LangChain](https://python.langchain.com/)
- [Chroma](https://docs.trychroma.com/)
- [Google Generative AI](https://ai.google.dev/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

## Notes

- The vector database is persisted in the project directory.
- Make sure your `.env` and PDF files are not tracked by git (see `.gitignore`).

