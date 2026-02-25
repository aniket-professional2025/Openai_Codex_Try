# Llama 2 Document Summarizer (8-bit)

This project provides a simple web app that:

1. Lets the user upload a `.txt` document.
2. Reads the text content from the file.
3. Sends the text to a Hugging Face **Llama 2** model (`meta-llama/Llama-2-7b-chat-hf`) loaded in **8-bit quantized** mode.
4. Returns a concise summary.

## Requirements

- Python 3.10+
- A machine with enough VRAM/RAM for Llama 2 7B in 8-bit mode
- Access to `meta-llama/Llama-2-7b-chat-hf` on Hugging Face
- Hugging Face token with model access

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Export your Hugging Face token:

```bash
export HUGGINGFACEHUB_API_TOKEN="your_hf_token"
```

## Run

```bash
python app.py
```

Then open: `http://localhost:7860`

## Optional environment variables

- `MODEL_NAME` (default: `meta-llama/Llama-2-7b-chat-hf`)
- `MAX_INPUT_CHARS` (default: `12000`)

## Notes

- Only `.txt` uploads are supported.
- Very long documents are truncated based on `MAX_INPUT_CHARS` before summarization.
