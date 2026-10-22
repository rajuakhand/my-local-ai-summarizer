# 🤖 Local AI Text Summarizer

An AI-powered text summarization web app built with Python, Streamlit, LangChain, and Ollama.

**Turn long text into concise, easy-to-understand summaries.**

[🚀 Live Demo](https://my-local-ai-summarizer-cetbdvs2ozk2mnvubxzeyj.streamlit.app/) · [💻 Source Code](https://github.com/rajuakhand/my-local-ai-summarizer)

## ✨ Features

- Summarize long text using an AI language model
- Simple and interactive Streamlit interface
- Built with Python and LangChain
- Uses Ollama for local language-model inference
- Easy to run and extend

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Ollama
- Llama 3.2 (1B)

## 🚀 Run Locally

Clone the repository:

```bash
git clone https://github.com/rajuakhand/my-local-ai-summarizer.git
cd my-local-ai-summarizer
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Make sure Ollama is installed and running, then download the model:

```bash
ollama pull llama3.2:1b
```

Start the application:

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 🌐 Live Demo

Try the deployed application:

https://my-local-ai-summarizer-cetbdvs2ozk2mnvubxzeyj.streamlit.app/

## 🗺️ Roadmap

- [ ] Add a one-click Copy Summary button
- [ ] Improve the UI and responsive layout
- [ ] Add Bengali and English language options
- [ ] Add sample text and word-count statistics
- [ ] Improve error handling and documentation

## 👨‍💻 About

This project demonstrates practical skills in Python, LLM integration, prompt engineering, and web app development.

If you find this project useful, consider giving the repository a ⭐.


This project is open source. A license can be added when one is chosen.
