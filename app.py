import os
import tempfile
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "12000"))


def load_model_and_tokenizer():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        token=token,
        torch_dtype=torch.float16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


MODEL = None
TOKENIZER = None


def get_model_and_tokenizer():
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        MODEL, TOKENIZER = load_model_and_tokenizer()
    return MODEL, TOKENIZER


def read_text_file(file_path: str) -> str:
    if not file_path:
        raise gr.Error("Please upload a .txt file.")

    path = Path(file_path)
    if path.suffix.lower() != ".txt":
        raise gr.Error("Only .txt files are supported.")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()

    if not content:
        raise gr.Error("The uploaded file is empty.")

    if len(content) > MAX_INPUT_CHARS:
        content = content[:MAX_INPUT_CHARS]

    return content


def build_prompt(text: str) -> str:
    return (
        "You are an expert document summarizer. "
        "Create a concise summary with:\n"
        "1) A one-paragraph overview\n"
        "2) 5 key bullet points\n"
        "3) Important action items (if any)\n\n"
        f"Document:\n{text}\n\nSummary:"
    )


def summarize_document(file_obj, max_new_tokens: int, temperature: float, top_p: float) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_obj)
        tmp_path = tmp.name

    try:
        text = read_text_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    model, tokenizer = get_model_and_tokenizer()
    prompt = build_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if not summary:
        return "No summary generated. Try increasing max_new_tokens."

    return summary


with gr.Blocks(title="Llama 2 Document Summarizer") as demo:
    gr.Markdown("# Llama 2 (8-bit) Document Summarizer")
    gr.Markdown(
        "Upload a `.txt` file. The app reads the content and uses a Hugging Face "
        "Llama 2 model with 8-bit quantization to produce a summary."
    )

    with gr.Row():
        file_input = gr.File(label="Upload .txt file", file_types=[".txt"], type="binary")

    with gr.Row():
        max_new_tokens = gr.Slider(64, 1024, value=256, step=32, label="Max new tokens")
        temperature = gr.Slider(0.0, 1.2, value=0.2, step=0.05, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")

    summarize_btn = gr.Button("Summarize")
    output = gr.Textbox(label="Summary", lines=16)

    summarize_btn.click(
        fn=summarize_document,
        inputs=[file_input, max_new_tokens, temperature, top_p],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
