# 🧠 Local AI Text Summarizer

A simple AI-powered text summarizer built with Python, Streamlit, LangChain, and Ollama.

## ✨ Features

- Summarizes text into three short sentences
- Runs locally using the Ollama language model
- Simple web interface powered by Streamlit
- No paid AI API key required

## 🛠️ Technologies

- Python
- Streamlit
- LangChain
- Ollama
- Llama 3.2 (1B)

## 📋 Requirements

- Python 3.12 or a compatible Python version
- Ollama installed
- The `llama3.2:1b` model downloaded

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/rajuakhand/my-local-ai-summarizer.git
cd my-local-ai-summarizer
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the AI model

```bash
ollama pull llama3.2:1b
```

### 5. Start the web app

Make sure Ollama is running, then run:

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

Open the local URL shown in the terminal, usually:

http://localhost:8501

## 🔒 Privacy

Text is processed using a local Ollama model. Avoid exposing private information when sharing your computer or using other services.

## 📄 License

This project is open source. A license can be added when one is chosen.
