import html
import io
import os
import random
import time
from datetime import datetime

import streamlit as st
from google import genai
from google.genai import types

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Preformatted,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors

    PDF_EXPORT_AVAILABLE = True
except ImportError:
    PDF_EXPORT_AVAILABLE = False


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Research & Text Summarizer Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

LANGUAGES = {
    "English": "English",
    "বাংলা (Bengali)": "Bengali",
    "Hindi": "Hindi",
    "Urdu": "Urdu",
    "Arabic": "Arabic",
    "German": "German",
    "French": "French",
    "Spanish": "Spanish",
    "Gujarati": "Gujarati",
    "Marathi": "Marathi",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "Chinese": "Simplified Chinese",
    "Russian": "Russian",
    "Portuguese": "Portuguese",
    "Italian": "Italian",
    "Turkish": "Turkish",
    "Indonesian": "Indonesian",
    "Persian": "Persian",
    "Nepali": "Nepali",
}

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
]

OUTPUT_MODES = {
    "Smart Summary": (
        "Write a coherent summary with the most important information."
    ),
    "Bullet Points": (
        "Summarize the information as clear, concise bullet points."
    ),
    "Key Takeaways": (
        "Extract the most important lessons and key takeaways."
    ),
    "Action Items": (
        "Extract actionable tasks. Include responsible people or "
        "deadlines only when explicitly present in the source."
    ),
    "Title Generator": (
        "Generate five concise, relevant title suggestions."
    ),
}

LENGTHS = {
    "Short": (
        "Keep the output brief, usually around 3-5 sentences."
    ),
    "Medium": (
        "Provide a balanced summary, usually around 6-10 sentences."
    ),
    "Detailed": (
        "Provide a detailed summary covering the main ideas, "
        "important facts, and relevant context."
    ),
}


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

DEFAULTS = {
    "input_text": "",
    "summary": "",
    "history": [],
    "last_language": "English",
    "last_mode": "Smart Summary",
    "last_model": MODELS[0],
    "last_length": "Medium",
    "last_uploaded_file": None,
    "chat_history": [],
    "chat_document": "",
    "chat_last_uploaded": None,
    "quiz_data": [],
    "quiz_answers": {},
    "flashcards": [],
    "slides_data": [],
    "ats_report": "",
    "solver_question": "",
    "solver_answer": "",
    "career_roadmap": "",
    "wellness_report": "",
    "eco_report": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# PROFESSIONAL STYLING (Scribbr-inspired Theme)
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }

    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }

    .hero-container h1 {
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .hero-container p {
        color: #94a3b8;
        font-size: 1.1rem;
        max-width: 700px;
        margin: 0 auto;
    }

    .trust-badges {
        display: flex;
        justify-content: center;
        gap: 24px;
        margin-top: 16px;
        font-size: 0.85rem;
        color: #cbd5e1;
        flex-wrap: wrap;
    }

    .trust-badge-item {
        background: rgba(255, 255, 255, 0.05);
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .service-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #6366f1;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 12px;
        font-size: 0.95rem;
        color: #e2e8f0;
    }

    .stButton button,
    .stDownloadButton button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 42px;
        transition: all 0.2s ease-in-out;
    }

    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 12px;
        border-radius: 12px;
    }

    div[data-baseweb="select"] > div {
        border-radius: 10px;
        background-color: #1e293b;
        color: white;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def get_api_key():
    """Read API key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.getenv("GEMINI_API_KEY", "")

    return key.strip() if key else ""


def classify_api_error(error):
    """Create a useful, user-friendly error message."""
    message = str(error)
    lower = message.lower()

    if any(term in lower for term in [
        "429", "resource_exhausted", "quota", "rate limit"
    ]):
        return (
            "API quota or rate limit reached. Wait for the quota "
            "to reset, or check your Google AI Studio usage limits."
        )

    if any(term in lower for term in [
        "503", "unavailable", "high demand", "overloaded"
    ]):
        return (
            "The AI server is temporarily unavailable or overloaded. "
            "Retry will be attempted automatically. If it continues, "
            "wait a few minutes and try again."
        )

    if any(term in lower for term in [
        "401", "403", "api key", "permission_denied",
        "unauthenticated"
    ]):
        return (
            "API authentication failed. Check GEMINI_API_KEY in "
            "Streamlit Settings > Secrets and verify that the key "
            "has access to the selected model."
        )

    if any(term in lower for term in [
        "404", "not_found", "not found"
    ]):
        return (
            "The selected model was not found or is unavailable "
            "for this API key. Try another supported model."
        )

    if any(term in lower for term in [
        "timeout", "timed out", "connection", "network"
    ]):
        return (
            "The request timed out or a network connection failed. "
            "Please retry."
        )

    return (
        "The request failed. Open Technical details below to inspect "
        "the original API error."
    )


def generate_with_retry(client, prompt, primary_model):
    """
    Retry temporary server errors and try the alternate model
    if the primary model remains unavailable.
    """
    fallback_model = next(
        (model for model in MODELS if model != primary_model),
        None,
    )

    models_to_try = [primary_model]

    if fallback_model:
        models_to_try.append(fallback_model)

    last_error = None

    for model_index, model_name in enumerate(models_to_try):
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=4096,
                    ),
                )

                result = response.text

                if result and result.strip():
                    return result.strip(), model_name

                raise RuntimeError(
                    "The model returned an empty response."
                )

            except Exception as error:
                last_error = error
                error_text = str(error).lower()

                retryable = any(
                    term in error_text
                    for term in [
                        "503",
                        "unavailable",
                        "high demand",
                        "overloaded",
                        "500",
                        "internal",
                        "timeout",
                        "timed out",
                        "429",
                        "resource_exhausted",
                    ]
                )

                if (
                    "429" in error_text
                    or "resource_exhausted" in error_text
                ):
                    raise

                if not retryable:
                    break

                if attempt < 2:
                    delay = min(2 ** (attempt + 1), 8)
                    delay += random.uniform(0, 0.5)
                    time.sleep(delay)

        if model_index < len(models_to_try) - 1:
            continue

    raise RuntimeError(
        "All configured models failed. Last error: "
        + str(last_error)
    )


def extract_uploaded_text(uploaded_file):
    """Extract text from TXT or PDF."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.getvalue().decode(
            "utf-8",
            errors="replace",
        )

    if filename.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError(
                "PDF support is missing. Install pypdf."
            )

        pdf_reader = PdfReader(
            io.BytesIO(uploaded_file.getvalue())
        )

        pages = [
            page.extract_text() or ""
            for page in pdf_reader.pages
        ]

        return "\n\n".join(pages)

    raise ValueError("Unsupported file type.")


def make_pdf(text):
    """Create a downloadable PDF summary."""
    if not PDF_EXPORT_AVAILABLE:
        raise RuntimeError(
            "PDF export requires reportlab."
        )

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=44,
        leftMargin=44,
        topMargin=44,
        bottomMargin=44,
    )

    styles = getSampleStyleSheet()
    story = []

    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    font_name = "Helvetica"

    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(
                    TTFont("SummaryUnicodeFont", font_path)
                )
                font_name = "SummaryUnicodeFont"
                break
            except Exception:
                pass

    styles["Normal"].fontName = font_name
    styles["Normal"].fontSize = 10
    styles["Normal"].leading = 15

    story.append(
        Paragraph(
            "AI Academic & Professional Summary",
            styles["Title"],
        )
    )
    story.append(Spacer(1, 16))

    for line in text.splitlines():
        safe_line = html.escape(line) or " "
        story.append(
            Paragraph(safe_line, styles["Normal"])
        )
        story.append(Spacer(1, 5))

    document.build(story)
    buffer.seek(0)

    return buffer.getvalue()


# --------------------------------------------------
# PROFESSIONAL HEADER & TRUST BANNER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero-container">
        <h1>🎓 AI Research & Text Summarizer Pro</h1>
        <p>
            Submit your work with confidence. Generate precise, multi-lingual academic summaries, 
            key takeaways, and structural insights powered by Google Gemini AI.
        </p>
        <div class="trust-badges">
            <span class="trust-badge-item">🔒 100% Academic Integrity</span>
            <span class="trust-badge-item">⚡ Instant 1-Click Processing</span>
            <span class="trust-badge-item">🌍 20+ Languages Supported</span>
            <span class="trust-badge-item">⭐ Enterprise Gemini AI</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Editing & AI Settings")

    language = st.selectbox(
        "Output language",
        list(LANGUAGES.keys()),
    )

    length = st.selectbox(
        "Summary length",
        list(LENGTHS.keys()),
        index=1,
    )

    output_mode = st.selectbox(
        "Output style",
        list(OUTPUT_MODES.keys()),
    )

    model_name = st.selectbox(
        "Primary AI model",
        MODELS,
        index=0,
        help=(
            "If the primary model has temporary server errors, "
            "the app will try the alternate configured model."
        ),
    )

    custom_instructions = st.text_area(
        "Custom instructions (optional)",
        placeholder=(
            "Example: Focus on research methodology and conclusion."
        ),
        height=100,
        max_chars=1000,
    )

    st.divider()
    st.subheader("📚 Professional Services")
    st.markdown(
        """
        - **Thesis & Dissertation Editing**
        - **Research Paper Structure Check**
        - **Clarity & Argument Polish**
        - **Automated Citation & Formatting**
        """
    )
    st.caption("Secured by Streamlit & Gemini API.")


# --------------------------------------------------
# INPUT SECTION (Error-Free State Management)
# --------------------------------------------------

left_col, right_col = st.columns(
    [1.1, 0.9],
    gap="large",
)

with left_col:
    st.subheader("📝 Input Document / Paper")

    uploaded_file = st.file_uploader(
        "Upload TXT or PDF research document",
        type=["txt", "pdf"],
        help="Scanned PDFs without a text layer may need OCR.",
    )

    if uploaded_file is not None:
        if st.session_state.last_uploaded_file != uploaded_file.name:
            try:
                uploaded_text = extract_uploaded_text(uploaded_file)
                if uploaded_text:
                    st.session_state.input_text = uploaded_text
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.success(f"Successfully loaded {uploaded_file.name}")
                    st.rerun()
                else:
                    st.warning("No readable text was found in this file.")
            except Exception as error:
                st.error(f"File processing failed: {error}")

    # Safe text_area without conflicting key parameter, preventing StreamlitWidgetAlreadyInstantiatedError
    st.session_state.input_text = st.text_area(
        "Paste or edit your document text here",
        value=st.session_state.input_text,
        height=300,
        placeholder="Paste your thesis, research paper, dissertation chapter, or essay...",
    )
    
    input_text = st.session_state.input_text

    words = len(input_text.split())
    characters = len(input_text)

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Word Count", f"{words:,}")
    metric2.metric("Characters", f"{characters:,}")
    metric3.metric("Mode", "Academic Pro")

    generate_col, clear_col = st.columns(2)

    with generate_col:
        generate_clicked = st.button(
            "✨ Generate Summary",
            type="primary",
            use_container_width=True,
        )

    with clear_col:
        clear_clicked = st.button(
            "🗑️ Clear Input",
            use_container_width=True,
        )

    if clear_clicked:
        st.session_state.input_text = ""
        st.session_state.summary = ""
        st.session_state.last_uploaded_file = None
        st.rerun()


# --------------------------------------------------
# GENERATION ENGINE
# --------------------------------------------------

if generate_clicked:
    api_key = get_api_key()

    if not input_text.strip():
        st.warning(
            "Please enter text or upload a document first."
        )

    elif len(input_text) > 60000:
        st.warning(
            "Please reduce the input to 60,000 characters or less."
        )

    elif not api_key:
        st.error(
            "GEMINI_API_KEY is missing. Add it in "
            "Streamlit Settings > Secrets."
        )

    else:
        prompt = f"""
You are a professional academic editor and multilingual text summarization assistant.

TARGET LANGUAGE:
{LANGUAGES[language]}

OUTPUT STYLE:
{output_mode}

LENGTH:
{LENGTHS[length]}

STYLE INSTRUCTIONS:
- Preserve the meaning and important facts.
- Never invent information absent from the source.
- Use natural, clear academic language.
- If the source contains uncertainty, preserve that uncertainty.
- Return the requested output in the target language.
- Do not add unrelated commentary.

CUSTOM INSTRUCTIONS:
{custom_instructions.strip() or "None"}

SOURCE TEXT:
<input>
{input_text}
</input>
"""

        try:
            with st.spinner(
                "🤖 Processing academic summary with Gemini AI..."
            ):
                client = genai.Client(api_key=api_key)

                generated_text, actual_model = generate_with_retry(
                    client,
                    prompt,
                    model_name,
                )

            st.session_state.summary = generated_text
            st.session_state.last_language = language
            st.session_state.last_mode = output_mode
            st.session_state.last_model = actual_model
            st.session_state.last_length = length

            history_item = {
                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "language": language,
                "mode": output_mode,
                "length": length,
                "model": actual_model,
                "source_preview": input_text[:180],
                "text": generated_text,
            }

            st.session_state.history.insert(
                0,
                history_item,
            )

            st.session_state.history = (
                st.session_state.history[:20]
            )

            st.success(
                f"Summary generated successfully using {actual_model}."
            )

        except Exception as error:
            st.error(classify_api_error(error))

            with st.expander("Technical error details"):
                st.code(str(error))


# --------------------------------------------------
# OUTPUT SECTION
# --------------------------------------------------

with right_col:
    st.subheader("📄 Summary & Analysis Output")

    if st.session_state.summary:
        summary = st.session_state.summary

        st.caption(
            f"Language: {st.session_state.last_language} "
            f"| Style: {st.session_state.last_mode} "
            f"| Model: {st.session_state.last_model}"
        )

        if st.session_state.last_language in {
            "Arabic",
            "Urdu",
            "Persian",
        }:
            safe_summary = html.escape(summary)

            st.markdown(
                f"""
                <div dir="rtl" style="
                    white-space: pre-wrap;
                    overflow-wrap: anywhere;
                    background: rgba(30,41,59,0.8);
                    border: 1px solid rgba(255,255,255,0.15);
                    border-radius: 14px;
                    padding: 18px;
                    line-height: 1.9;
                ">{safe_summary}</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="glass-card">',
                unsafe_allow_html=True,
            )
            st.write(summary)
            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        st.download_button(
            "⬇️ Download as TXT",
            data=summary,
            file_name="ai_academic_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

        if PDF_EXPORT_AVAILABLE:
            try:
                pdf_bytes = make_pdf(summary)

                st.download_button(
                    "📄 Download as PDF Report",
                    data=pdf_bytes,
                    file_name="ai_academic_summary.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            except Exception as error:
                st.warning(
                    "PDF export failed. TXT download is still available."
                )
                with st.expander("PDF export details"):
                    st.code(str(error))
        else:
            st.info(
                "To enable PDF export, add reportlab to requirements.txt."
            )

        safe_copy_text = html.escape(summary)

        st.components.v1.html(
            f"""
            <div style="
                font-family: sans-serif;
                background: #1e293b;
                padding: 12px;
                border-radius: 12px;
                color: white;
                border: 1px solid rgba(255,255,255,0.1);
            ">
                <textarea id="summary-copy-area" style="
                    width: 100%;
                    height: 120px;
                    box-sizing: border-box;
                    padding: 10px;
                    border-radius: 8px;
                    background: #0f172a;
                    color: #ffffff;
                    border: 1px solid #475569;
                    font-size: 14px;
                    line-height: 1.5;
                ">{safe_copy_text}</textarea>

                <button id="summary-copy-button" style="
                    margin-top: 10px;
                    width: 100%;
                    padding: 10px;
                    border: 0;
                    border-radius: 8px;
                    background: #6366f1;
                    color: white;
                    font-weight: bold;
                    cursor: pointer;
                ">📋 Copy Summary to Clipboard</button>

                <div id="summary-copy-status" style="
                    padding-top: 6px;
                    font-size: 12px;
                    color: #38bdf8;
                "></div>
            </div>

            <script>
            const copyArea = document.getElementById("summary-copy-area");
            const copyButton = document.getElementById("summary-copy-button");
            const copyStatus = document.getElementById("summary-copy-status");

            copyButton.addEventListener("click", async () => {{
                try {{
                    await navigator.clipboard.writeText(copyArea.value);
                    copyStatus.textContent = "Copied successfully!";
                }} catch (error) {{
                    copyArea.focus();
                    copyArea.select();
                    const copied = document.execCommand("copy");
                    copyStatus.textContent = copied ? "Copied successfully!" : "Please copy manually.";
                }}
            }});
            </script>
            """,
            height=220,
        )

        if st.button(
            "🆕 Start New Summary",
            use_container_width=True,
        ):
            st.session_state.input_text = ""
            st.session_state.summary = ""
            st.session_state.last_uploaded_file = None
            st.rerun()

    else:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Ready to Transform Your Document</h3>
                <p>
                    Paste your thesis, essay, or research paper on the left, select your formatting and output style, 
                    and let our professional AI assistant generate precise insights instantly.
                </p>
                <div class="service-box">
                    <strong>💡 Pro Tip:</strong> Use custom instructions in the sidebar to target specific research questions or methodology reviews.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "If a model is busy, the app automatically retries and switches to the backup model."
        )


# --------------------------------------------------
# SUMMARY HISTORY
# --------------------------------------------------

st.divider()
st.subheader("🕘 Session History & Archive")

history = st.session_state.history

if history:
    history_labels = [
        (
            f"{item['timestamp']} | "
            f"{item['language']} | "
            f"{item['mode']} | "
            f"{item['text'][:45].replace(chr(10), ' ')}"
        )
        for item in history
    ]

    selected_history_index = st.selectbox(
        "Choose a previous summary from archive",
        range(len(history)),
        format_func=lambda index: history_labels[index],
    )

    selected_history = history[selected_history_index]

    history_col1, history_col2 = st.columns(2)

    with history_col1:
        if st.button(
            "↩️ Restore Selected Summary",
            use_container_width=True,
        ):
            st.session_state.summary = selected_history["text"]
            st.session_state.last_language = (
                selected_history["language"]
            )
            st.session_state.last_mode = selected_history["mode"]
            st.session_state.last_length = selected_history["length"]
            st.session_state.last_model = selected_history["model"]
            st.rerun()

    with history_col2:
        if st.button(
            "🧹 Clear Archive",
            use_container_width=True,
        ):
            st.session_state.history = []
            st.rerun()

    with st.expander("Preview selected archive item"):
        st.caption(
            f"Model: {selected_history['model']} "
            f"| Length: {selected_history['length']}"
        )
        st.write(selected_history["text"])

else:
    st.info(
        "Your generated summaries will appear here for quick reference during your active session."
    )


# --------------------------------------------------
# CHAT WITH DOCUMENT (Document Q&A)
# --------------------------------------------------

st.divider()
st.subheader("💬 Chat with Document")
st.caption("Ask questions about your uploaded document and get instant answers.")

chat_col1, chat_col2 = st.columns([1, 1], gap="large")

with chat_col1:
    st.markdown("**Upload Document for Chat**")
    chat_uploaded_file = st.file_uploader(
        "Upload TXT or PDF to chat with",
        type=["txt", "pdf"],
        key="chat_uploader",
        help="Upload a document to start chatting with it.",
    )

    if chat_uploaded_file is not None:
        if st.session_state.chat_last_uploaded != chat_uploaded_file.name:
            try:
                chat_doc_text = extract_uploaded_text(chat_uploaded_file)
                if chat_doc_text:
                    st.session_state.chat_document = chat_doc_text
                    st.session_state.chat_last_uploaded = chat_uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"Loaded {chat_uploaded_file.name} for chat")
                    st.rerun()
                else:
                    st.warning("No readable text found in this file.")
            except Exception as error:
                st.error(f"File processing failed: {error}")

    if st.session_state.chat_document:
        chat_words = len(st.session_state.chat_document.split())
        st.metric("Document Size", f"{chat_words:,} words")
        
        with st.expander("View Document Preview"):
            preview = st.session_state.chat_document[:1000]
            st.text(preview + ("..." if len(st.session_state.chat_document) > 1000 else ""))
    else:
        st.info("Upload a document to start chatting.")

with chat_col2:
    st.markdown("**Chat Interface**")
    
    if not st.session_state.chat_document:
        st.info("Please upload a document first to ask questions.")
    else:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_history:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"**🧑 You:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖 AI:** {msg['content']}")
                    st.markdown("---")
            else:
                st.caption("No messages yet. Ask your first question!")
        
        # Question input
        user_question = st.text_input(
            "Ask a question about the document:",
            placeholder="e.g., What are the main conclusions?",
            key="chat_question",
        )
        
        chat_btn_col1, chat_btn_col2 = st.columns(2)
        with chat_btn_col1:
            ask_clicked = st.button("💬 Ask Question", type="primary", use_container_width=True)
        with chat_btn_col2:
            clear_chat_clicked = st.button("🗑️ Clear Chat", use_container_width=True)
        
        if clear_chat_clicked:
            st.session_state.chat_history = []
            st.rerun()
        
        if ask_clicked and user_question.strip():
            api_key = get_api_key()
            if not api_key:
                st.error("GEMINI_API_KEY is missing. Add it in Streamlit Settings > Secrets.")
            else:
                try:
                    with st.spinner("🤖 Thinking..."):
                        client = genai.Client(api_key=api_key)
                        
                        chat_prompt = f"""
You are an expert document analyst. Answer questions based ONLY on the provided document context.

RULES:
- Answer ONLY using information from the document. Do not use prior knowledge.
- If the answer is not found in the document, say "I cannot find this information in the document."
- Be precise and cite specific details when relevant.
- Keep answers clear and concise.

DOCUMENT CONTEXT:
{st.session_state.chat_document[:50000]}

QUESTION:
{user_question}

ANSWER:
"""
                        
                        generated_answer, actual_model = generate_with_retry(
                            client,
                            chat_prompt,
                            model_name,
                        )
                        
                        # Add to history
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": user_question,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": generated_answer,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        
                        st.rerun()
                        
                except Exception as error:
                    st.error(classify_api_error(error))
                    with st.expander("Technical error details"):
                        st.code(str(error))
        elif ask_clicked and not user_question.strip():
            st.warning("Please enter a question.")

# --------------------------------------------------
# AI PRESENTATION MAKER (Slide Generator)
# --------------------------------------------------

st.divider()
st.subheader("🎞️ AI Presentation Maker")
st.caption("Turn your document into professional slides with speaker notes instantly.")

pcol1, pcol2 = st.columns([1, 1], gap="large")

with pcol1:
    st.markdown("**Presentation Settings**")
    slide_source = st.radio(
        "Source Document",
        ["Use current input document", "Use chat document", "Paste text manually"],
        key="slide_source"
    )
    
    if slide_source == "Paste text manually":
        slide_manual_text = st.text_area(
            "Paste text for presentation",
            height=200,
            key="slide_manual"
        )
        slide_source_text = slide_manual_text
    elif slide_source == "Use chat document":
        slide_source_text = st.session_state.chat_document
        if slide_source_text:
            st.info(f"Using chat document ({len(slide_source_text.split()):,} words)")
        else:
            st.warning("No chat document uploaded. Upload in Chat section first.")
    else:
        slide_source_text = st.session_state.input_text
        if slide_source_text:
            st.info(f"Using input document ({len(slide_source_text.split()):,} words)")
        else:
            st.warning("No input document available. Upload/paste in Input section first.")
    
    slide_count = st.slider("Number of Slides", 5, 12, 8)
    slide_audience = st.selectbox(
        "Target Audience",
        ["Academic/Lecture", "Student Presentation", "Business/Meeting", "Seminar/Conference"],
        key="slide_audience"
    )
    slide_style = st.selectbox(
        "Presentation Style",
        ["Concise & Clear", "Detailed with Explanation", "Minimal Bullet Points"],
        key="slide_style"
    )
    
    gen_slides_btn = st.button("🎞️ Generate Presentation", type="primary", use_container_width=True)
    
    if gen_slides_btn:
        api_key = get_api_key()
        if not api_key:
            st.error("GEMINI_API_KEY is missing.")
        elif not slide_source_text.strip():
            st.warning("Please provide a document source first.")
        else:
            try:
                with st.spinner("🤖 Creating your presentation slides..."):
                    import json
                    client = genai.Client(api_key=api_key)
                    slide_prompt = f"""
You are an expert presentation designer. Create a professional {slide_count}-slide presentation from the document.

AUDIENCE: {slide_audience}
STYLE: {slide_style}
RULES:
- Create exactly {slide_count} slides
- Slide 1 should be Title Slide
- Include Introduction, Main Content, and Conclusion slides appropriately
- Each slide must have: slide_number, title, bullets (list of points), speaker_notes
- Keep bullets concise and impactful
- Speaker notes should explain what to say during presentation
- Content must be based ONLY on the document
- Format as JSON

DOCUMENT:
{slide_source_text[:45000]}

Return in this exact format:
[
  {{
    "slide_number": 1,
    "title": "Slide Title",
    "bullets": ["Point 1", "Point 2", "Point 3"],
    "speaker_notes": "What to say while presenting this slide..."
  }}
]
"""
                    resp, model_used = generate_with_retry(client, slide_prompt, model_name)
                    try:
                        start = resp.find('[')
                        end = resp.rfind(']') + 1
                        json_str = resp[start:end]
                        slides_data = json.loads(json_str)
                        st.session_state.slides_data = slides_data
                        st.session_state.slide_idx = 0
                        st.success(f"Generated {len(slides_data)} slides!")
                        st.rerun()
                    except:
                        st.error("Failed to parse slides. Please try again.")
                        st.code(resp)
            except Exception as error:
                st.error(classify_api_error(error))

with pcol2:
    st.markdown("**Slide Preview**")
    if st.session_state.slides_data:
        if 'slide_idx' not in st.session_state:
            st.session_state.slide_idx = 0
        
        total_s = len(st.session_state.slides_data)
        slide = st.session_state.slides_data[st.session_state.slide_idx]
        
        st.progress((st.session_state.slide_idx + 1) / total_s)
        st.caption(f"Slide {slide['slide_number']} of {total_s}")
        
        st.markdown(f"""
        <div style="background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); border: 2px solid #6366f1; border-radius: 14px; padding: 30px; min-height: 280px; color: white;">
            <h2 style="color: #a5b4fc; margin-bottom: 24px; font-size: 1.8rem;">{html.escape(str(slide['title']))}</h2>
            <ul style="font-size: 1.1rem; line-height: 1.8;">
        """, unsafe_allow_html=True)
        
        for bullet in slide.get('bullets', []):
            st.markdown(f"<li style='margin-bottom: 10px;'>{html.escape(str(bullet))}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
        
        with st.expander("🎤 Speaker Notes"):
            st.write(slide.get('speaker_notes', 'No notes available.'))
        
        nav_s1, nav_s2, nav_s3 = st.columns(3)
        with nav_s1:
            if st.button("⬅️ Previous", use_container_width=True):
                if st.session_state.slide_idx > 0:
                    st.session_state.slide_idx -= 1
                    st.rerun()
        with nav_s2:
            pass
        with nav_s3:
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.slide_idx < total_s - 1:
                    st.session_state.slide_idx += 1
                    st.rerun()
        
        md_content = f"# Presentation\n\n"
        for s in st.session_state.slides_data:
            md_content += f"## Slide {s['slide_number']}: {s['title']}\n\n"
            for b in s.get('bullets', []):
                md_content += f"- {b}\n"
            md_content += f"\n**Speaker Notes:**\n{s.get('speaker_notes', '')}\n\n---\n\n"
        
        st.download_button(
            "📥 Download as Markdown",
            data=md_content,
            file_name="ai_presentation.md",
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.info("Generate a presentation first to preview slides here.")


st.divider()

# --------------------------------------------------
# AI ATS RESUME VS JOB DESCRIPTION MATCHER
# --------------------------------------------------

st.subheader("🎯 AI ATS Resume vs Job Description Matcher")
st.caption("Check your resume's ATS compatibility against any job description and get actionable suggestions.")

ats_c1, ats_c2 = st.columns([1, 1], gap="large")

with ats_c1:
    st.markdown("### 1) Upload Resume")
    resume_up = st.file_uploader("Upload Resume (TXT or PDF)", type=["txt", "pdf"], key="ats_resume")
    resume_txt = ""
    if resume_up is not None:
        try:
            resume_txt = extract_uploaded_text(resume_up)
            if resume_txt.strip():
                st.success(f"Loaded {resume_up.name}")
                with st.expander("Resume Preview"):
                    st.text(resume_txt[:700] + ("..." if len(resume_txt) > 700 else ""))
            else:
                st.warning("No readable text found.")
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.markdown("### 2) Job Description")
    jd_mode = st.radio("Provide Job Description", ["Paste text", "Upload file (TXT/PDF)"], key="jd_mode")
    jd_txt = ""
    if jd_mode == "Paste text":
        jd_txt = st.text_area("Paste Job Description", height=220, placeholder="Job title, requirements, responsibilities, qualifications...", key="jd_area")
    else:
        jd_up = st.file_uploader("Upload JD (TXT or PDF)", type=["txt", "pdf"], key="jd_up")
        if jd_up is not None:
            try:
                jd_txt = extract_uploaded_text(jd_up)
                if jd_txt.strip():
                    st.success(f"Loaded {jd_up.name}")
                    with st.expander("JD Preview"):
                        st.text(jd_txt[:600] + ("..." if len(jd_txt) > 600 else ""))
                else:
                    st.warning("No readable text found.")
            except Exception as e:
                st.error(f"Error: {e}")
    
    ats_run = st.button("🔍 Analyze ATS Match", type="primary", use_container_width=True)
    
    if ats_run:
        api_key = get_api_key()
        if not api_key:
            st.error("GEMINI_API_KEY missing. Add in Streamlit Secrets.")
        elif not resume_txt.strip():
            st.warning("Upload your resume first.")
        elif not jd_txt.strip():
            st.warning("Provide job description first.")
        else:
            try:
                with st.spinner("Analyzing against ATS criteria..."):
                    import re, json
                    client = genai.Client(api_key=api_key)
                    prompt = f"""You are an expert ATS analyst. Compare this resume against the job description.

RESUME:
{resume_txt[:38000]}

JOB DESCRIPTION:
{jd_txt[:38000]}

Return a detailed report in this exact structure:

## ATS MATCH SCORE
Give score 0-100 with 1 line reasoning

## OVERALL VERDICT
Excellent/Good/Moderate/Poor + explanation

## MATCHED KEYWORDS
Top matched hard/soft skills (8-12)

## MISSING KEYWORDS
Critical missing keywords (8-12), prioritize by importance

## HARD SKILLS GAP
Specific technical skills missing

## SOFT SKILLS GAP
Relevant soft skills missing

## STRENGTHS
Top 4 strengths

## WEAKNESSES
Top 4 weaknesses

## ATS OPTIMIZATION TIPS
5 actionable tips (concrete rewrites suggested)

## QUICK WINS (Do in 30 min)
3 immediate fixes

Be specific, practical, and honest."""
                    res, used = generate_with_retry(client, prompt, model_name)
                    st.session_state.ats_report = res
                    st.success("Analysis complete!")
                    st.rerun()
            except Exception as e:
                st.error(classify_api_error(e))

with ats_c2:
    st.markdown("### Analysis Report")
    if st.session_state.get("ats_report"):
        rep = st.session_state.ats_report
        import re
        m = re.search(r'ATS MATCH SCORE[^\d]*(\d+)', rep, re.I)
        if m:
            sc = int(m.group(1))
            if sc >= 80: st.success(f"🎉 ATS Score: {sc}/100 — Excellent")
            elif sc >= 65: st.info(f"👍 ATS Score: {sc}/100 — Good")
            elif sc >= 50: st.warning(f"⚠️ ATS Score: {sc}/100 — Moderate")
            else: st.error(f"❌ ATS Score: {sc}/100 — Needs Work")
            st.progress(sc/100)
        st.markdown(rep)
        st.download_button("📥 Download Report (TXT)", data=rep, file_name=f"ats_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", use_container_width=True)
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.ats_report = ""
            st.rerun()
    else:
        st.markdown("""
        <div class="glass-card">
            <h4>What you'll get</h4>
            <ul>
                <li>ATS Match Score (0-100)</li>
                <li>Matched vs Missing Keywords</li>
                <li>Skills Gap Analysis</li>
                <li>Tailored Fix Suggestions</li>
                <li>Quick Wins to Apply Today</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# AI PROBLEM SOLVER & SMART TUTOR
# --------------------------------------------------

st.subheader("🧠 AI Problem Solver & Smart Tutor")
st.caption("Get step-by-step, mathematical, conceptual or analytical solutions in the simplest way possible.")

ps_c1, ps_c2 = st.columns([1, 1], gap="large")

with ps_c1:
    st.markdown("### Problem Input")
    ps_mode = st.radio("Input Method", ["Paste Question", "Upload File (TXT/PDF)"], key="ps_mode")
    ps_qtext = ""
    if ps_mode == "Paste Question":
        ps_qtext = st.text_area("Enter your question/problem here", height=200, placeholder="Paste math, physics, chemistry, or any question...\nYou can also paste multiple questions.", key="ps_qarea")
        st.session_state.solver_question = ps_qtext
    else:
        ps_f = st.file_uploader("Upload question file", type=["txt", "pdf"], key="ps_file")
        if ps_f is not None:
            try:
                ps_qtext = extract_uploaded_text(ps_f)
                if ps_qtext.strip():
                    st.success(f"Loaded {ps_f.name}")
                    st.session_state.solver_question = ps_qtext
                    with st.expander("Preview"):
                        st.text(ps_qtext[:700] + ("..." if len(ps_qtext) > 700 else ""))
                else:
                    st.warning("Empty file")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("### Solution Settings")
    ps_subject = st.selectbox("Subject", ["Mathematics", "Physics", "Chemistry", "Biology", "Economics", "Statistics", "Logic/Reasoning", "Computer Science", "General/Other"], key="ps_subj")
    ps_goal = st.selectbox("Solution Type", ["Step-by-Step Solution", "Mathematical Derivation", "Conceptual Explanation", "Analytical Reasoning", "Exam-Oriented Answer", "Explain Like I'm a Beginner"], key="ps_goal")
    ps_level = st.select_slider("Explanation Level", options=["Beginner", "Intermediate", "Advanced"], value="Beginner")
    ps_alt = st.checkbox("Also show alternative method", value=True)
    ps_prac = st.checkbox("Generate 2 practice questions based on this", value=True)
    
    ps_btn = st.button("🚀 Solve Problem", type="primary", use_container_width=True)
    
    if ps_btn:
        api_key = get_api_key()
        qin = ps_qtext if ps_qtext.strip() else st.session_state.get("solver_question", "")
        if not api_key:
            st.error("GEMINI_API_KEY missing.")
        elif not qin.strip():
            st.warning("Enter or upload a question first.")
        else:
            try:
                with st.spinner("Thinking through this problem..."):
                    client = genai.Client(api_key=api_key)
                    prompt = f"""You are an expert tutor who explains complex problems clearly with logic.

SUBJECT: {ps_subject}
GOAL: {ps_goal}
LEVEL: {ps_level}
SHOW ALTERNATIVE METHOD: {ps_alt}
GENERATE PRACTICE: {ps_prac}

QUESTION:
{qin[:45000]}

Follow these rules strictly:
- Give a crystal clear answer tailored to {ps_level}
- Use step by step breakdown with numbers/logic
- Show formulas used if any
- For math: show each calculation clearly
- For theory: explain with reasoning
- Use simple words first, then precise terms
- If diagram logic helps, describe in text
- Be accurate, don't guess

Structure your response as:

## PROBLEM STATEMENT
[Restate clearly]

## SOLUTION (STEP BY STEP)
[Numbered steps with explanation]

## FINAL ANSWER
[Clear final answer highlighted]

## KEY CONCEPTS USED
[Bulleted list]

## COMMON MISTAKES TO AVOID
[2-3 points]

{"## ALTERNATIVE METHOD" if ps_alt else ""}
{"[If alternative exists, show another approach with steps]" if ps_alt else ""}

{"## PRACTICE QUESTIONS FOR YOU" if ps_prac else ""}
{"[Give 2 new questions on same topic with answers at end: 'ANSWERS FOR PRACTICE']" if ps_prac else ""}

Return in clean markdown."""
                    ans, used = generate_with_retry(client, prompt, model_name)
                    st.session_state.solver_answer = ans
                    st.success("Solved!")
                    st.rerun()
            except Exception as e:
                st.error(classify_api_error(e))

with ps_c2:
    st.markdown("### Solution")
    if st.session_state.get("solver_answer"):
        sol = st.session_state.solver_answer
        st.markdown(sol)
        st.download_button("📥 Download Solution (TXT)", data=sol, file_name=f"ai_problem_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", use_container_width=True)
        if st.button("🧹 Clear Solution", use_container_width=True):
            st.session_state.solver_answer = ""
            st.rerun()
    else:
        st.markdown("""
        <div class="glass-card">
            <h4>How this works</h4>
            <ul>
                <li>Paste any question or upload file</li>
                <li>Pick subject, goal & level</li>
                <li>Get step-by-step solution</li>
                <li>See alternative method & practice questions</li>
                <li>Perfect for homework, exam prep & self-study</li>
            </ul>
            <div class="service-box">
                <strong>Tip:</strong> For math problems, paste full question with numbers for best results
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# AI CAREER ROADMAP GENERATOR (FROM ZERO TO GOAL)
# --------------------------------------------------

st.subheader("🗺️ AI Career Roadmap Generator (From Zero to Goal)")
st.caption("Get a complete A to Z roadmap for any career - books, timeline, skills, internships, jobs & more.")

cr_c1, cr_c2 = st.columns([1, 1], gap="large")

with cr_c1:
    st.markdown("### Your Goal")
    career_list = ["Doctor (MBBS/Medical)", "Engineer (CSE)", "Engineer (Civil)", "Engineer (EEE/Mechanical)", "Lawyer (LLB)", "Chartered Accountant (CA)", "Data Scientist/AI", "Business/Entrepreneurship", "Pilot", "Nurse/Pharmacy", "Banking/Finance", "Government Job (BCS/PSC)", "Freelancer/Remote Work", "Other (Custom)"]
    career_sel = st.selectbox("Select Career Goal", career_list, key="career_sel")
    
    if career_sel == "Other (Custom)":
        career_custom = st.text_input("Enter your desired career", placeholder="e.g. UX Designer, Cybersecurity, Architecture", key="career_cust")
        target_career = career_custom
    else:
        target_career = career_sel
    
    st.markdown("### Your Current Situation")
    cr_level = st.selectbox("Current Status", ["Class 9-10", "HSC/A-Level", "After HSC", "Undergraduate Student", "Graduate/Job Holder", "Complete Beginner (Zero Knowledge)"], key="cr_level")
    cr_time = st.selectbox("Time Available Daily", ["30-60 min", "1-2 hours", "2-3 hours", "3-4 hours", "4-6 hours", "Full Time Focused"], key="cr_time")
    cr_goal_time = st.selectbox("Target Timeline", ["1 Year", "2 Years", "3 Years", "4-5 Years", "As Fast as Possible", "Flexible"], key="cr_goal_t")
    cr_country = st.text_input("Country/Location (optional)", placeholder="e.g. Bangladesh, India, Pakistan, USA", key="cr_country")
    
    cr_gen = st.button("🗺️ Generate Full Roadmap", type="primary", use_container_width=True)
    
    if cr_gen:
        api_key = get_api_key()
        if not api_key:
            st.error("GEMINI_API_KEY missing.")
        elif not target_career.strip():
            st.warning("Enter/select a career goal.")
        else:
            try:
                with st.spinner("Creating your complete A to Z roadmap..."):
                    client = genai.Client(api_key=api_key)
                    prompt = f"""You are a practical career coach who creates realistic, actionable roadmaps for students starting from zero.

CREATE A COMPLETE A TO Z ROADMAP FOR: {target_career}
CURRENT STATUS: {cr_level}
DAILY TIME: {cr_time}
TARGET TIMELINE: {cr_goal_time}
LOCATION: {cr_country or 'General'}

Make it brutally practical and realistic. Cover EVERYTHING a student needs.

Structure with these exact sections:

# COMPLETE ROADMAP: FROM ZERO TO {target_career.upper()}

## 1) OVERVIEW
Quick summary of this career, demand, scope, realistic outlook

## 2) MINDSET & FIRST STEPS (WEEK 1)
What to do immediately

## 3) PHASE WISE ROADMAP
Break into Phase 1, Phase 2, Phase 3... with timeline, goals, tasks

## 4) BOOKS & RESOURCES
Must read books (name + free alternatives), websites, YouTube channels, apps

## 5) SKILLS TO LEARN
Core skills + Extra/High-value skills with order to learn

## 6) EXAM/ADMISSION GUIDE
Entrance exams, eligibility, preparation strategy if applicable

## 7) PRACTICAL PROJECTS
Projects to build portfolio step by step

## 8) INTERNSHIP GUIDE
When to start, where to find, how to apply, what to learn

## 9) NETWORKING & LINKEDIN
How to build profile, connect, get opportunities

## 10) JOB SEARCH STRATEGY (AFTER GRADUATION)
CV/Resume, apply strategy, interview prep, where to apply

## 11) MONTHLY/WEEKLY TIME PLAN
Realistic daily split based on {cr_time}

## 12) COST ESTIMATE
Free vs paid (realistic)

## 13) SALARY REALITY
Entry level to experienced range (realistic for market)

## 14) COMMON MISTAKES TO AVOID
Must know

## 15) 90-DAY ACTION PLAN
Exact actions for first 90 days

Be country-aware if location given. Prioritize FREE resources. Focus on practical over theory.
"""
                    res, used = generate_with_retry(client, prompt, model_name)
                    st.session_state.career_roadmap = res
                    st.success("Roadmap generated!")
                    st.rerun()
            except Exception as e:
                st.error(classify_api_error(e))

with cr_c2:
    st.markdown("### Your Roadmap")
    if st.session_state.get("career_roadmap"):
        rm = st.session_state.career_roadmap
        st.markdown(rm)
        st.download_button("📥 Download Full Roadmap (TXT)", data=rm, file_name=f"Career_Roadmap_{target_career.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt", use_container_width=True)
        if st.button("🔄 Generate New Roadmap", use_container_width=True):
            st.session_state.career_roadmap = ""
            st.rerun()
    else:
        st.markdown("""
        <div class="glass-card">
            <h4>From Zero to Goal in One Click</h4>
            <p>Get a complete roadmap covering:</p>
            <ul>
                <li>📚 Books to read & free resources</li>
                <li>⏰ Hourly/Daily study plan</li>
                <li>🛠️ Skills to master step by step</li>
                <li>💼 Internships, networking & LinkedIn</li>
                <li>🎯 Exam/admission guide</li>
                <li>🚀 Job search after completion</li>
                <li>💰 Realistic salary info</li>
                <li>⚡ First 90-Day Action Plan</li>
            </ul>
            <div class="service-box">
                <strong>Example:</strong> "Doctor", "CSE Engineer", "Lawyer" etc - works for any field
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# AI STUDENT FITNESS, MENTAL & WELLNESS SELF-CHECKER
# --------------------------------------------------

st.subheader("❤️ AI Student Fitness, Mental & Wellness Self-Checker")
st.caption("Quick self-check for your physical, mental & overall wellness. Get safe, practical advice tailored for students.")

wl_c1, wl_c2 = st.columns([1, 1], gap="large")

with wl_c1:
    st.markdown("### 👤 Basic Info")
    wl_age = st.number_input("Age", min_value=10, max_value=40, value=18)
    wl_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    wl_height = st.number_input("Height (cm)", min_value=120, max_value=210, value=170)
    wl_weight = st.number_input("Weight (kg)", min_value=35, max_value=150, value=65)
    
    try:
        bmi = wl_weight / ((wl_height/100)**2)
        if bmi < 18.5:
            st.info(f"BMI: {bmi:.1f} — Underweight")
        elif bmi < 25:
            st.success(f"BMI: {bmi:.1f} — Normal weight")
        elif bmi < 30:
            st.warning(f"BMI: {bmi:.1f} — Overweight")
        else:
            st.error(f"BMI: {bmi:.1f} — Obese")
    except:
        pass
    
    st.markdown("### 🏃 Physical Habits")
    wl_sleep = st.slider("Average Sleep (hours/night)", 3.0, 10.0, 7.0, 0.5)
    wl_water = st.slider("Water intake (glasses/day)", 1, 12, 6)
    wl_walk = st.selectbox("Daily Activity", ["Mostly sitting (all day)", "Light walk 1-2 days", "Walk 3-4 days/week", "Active most days", "Very active daily"])
    wl_meal = st.selectbox("Eating Habits", ["Mostly junk/fast food", "Mixed", "Home cooked mostly", "Balanced & nutritious"])
    wl_screen = st.slider("Screen time (hours/day)", 1, 14, 7)
    
    st.markdown("### 🧠 Mental Wellness")
    wl_stress = st.select_slider("Stress Level", options=["Low", "Mild", "Moderate", "High", "Very High"])
    wl_mood = st.select_slider("Overall Mood (last 2 weeks)", options=["Mostly good", "Sometimes low", "Often low", "Frequently anxious/sad", "Very distressed"])
    wl_focus = st.select_slider("Focus/Concentration", options=["Good", "Slightly affected", "Often distracted", "Hard to focus", "Very poor"])
    wl_sleep_qual = st.select_slider("Sleep Quality", options=["Restful", "Mostly fine", "Disturbed", "Poor", "Very poor"])
    wl_social = st.select_slider("Social Connection", options=["Active", "Okay", "Isolated at times", "Mostly alone", "Lonely often"])
    
    st.markdown("### ⚠️ Red Flags (Check if any)")
    wl_headache = st.checkbox("Frequent headaches")
    wl_chest = st.checkbox("Chest discomfort on exertion")
    wl_breath = st.checkbox("Shortness of breath unusually")
    wl_dizzy = st.checkbox("Dizziness/fainting spells")
    wl_anxiety = st.checkbox("Panic-like symptoms")
    wl_insomnia = st.checkbox("Severe insomnia")
    
    wl_check = st.button("🩺 Get Wellness Report", type="primary", use_container_width=True)
    
    if wl_check:
        api_key = get_api_key()
        if not api_key:
            st.error("GEMINI_API_KEY missing.")
        else:
            try:
                with st.spinner("Analyzing your fitness & mental wellness..."):
                    red = []
                    if wl_headache: red.append("Frequent headaches")
                    if wl_chest: red.append("Chest discomfort on exertion")
                    if wl_breath: red.append("Shortness of breath unusually")
                    if wl_dizzy: red.append("Dizziness/fainting spells")
                    if wl_anxiety: red.append("Panic-like symptoms")
                    if wl_insomnia: red.append("Severe insomnia")
                    
                    client = genai.Client(api_key=api_key)
                    prompt = f"""You are a student wellness advisor. Give safe, practical advice. DO NOT diagnose disease. Recommend doctor if serious.

BASIC: Age {wl_age}, Gender {wl_gender}, Height {wl_height}cm, Weight {wl_weight}kg, BMI ~{bmi:.1f}
PHYSICAL: Sleep {wl_sleep}h, Water {wl_water} glasses, Activity {wl_walk}, Meals {wl_meal}, Screen {wl_screen}h
MENTAL: Stress {wl_stress}, Mood {wl_mood}, Focus {wl_focus}, Sleep Quality {wl_sleep_qual}, Social {wl_social}
RED FLAGS: {', '.join(red) if red else 'None'}

Give report in this format:

## OVERALL WELLNESS SCORE
Physical /100 + Mental /100 + Overall /100 with short verdict

## PHYSICAL FITNESS ASSESSMENT
Strengths + Areas to improve

## MENTAL WELLNESS ASSESSMENT
Observations + gentle advice

## BMI & BODY AWARENESS
Simple explanation

## HOME WORKOUT PLAN (10-15 MIN)
Easy exercises for beginners (no gym needed)

## YOGA & BREATHING (FOR STUDENTS)
Simple asanas + pranayama (5-10 min)

## DAILY ROUTINE SUGGESTION
Wake up to sleep realistic plan

## WHEN TO SEE A DOCTOR
Clear list based on red flags

## 7-DAY ACTION PLAN
Day1 to Day7 specific tasks

## SAFETY DISCLAIMER
Must say: This is for awareness only, not a substitute for professional medical advice. Seek doctor for serious/red flag symptoms.

Keep advice student-friendly, affordable, realistic in hostel/home."""
                    res, used = generate_with_retry(client, prompt, model_name)
                    st.session_state.wellness_report = res
                    st.success("Report ready!")
                    st.rerun()
            except Exception as e:
                st.error(classify_api_error(e))

with wl_c2:
    st.markdown("### Your Wellness Report")
    if st.session_state.get("wellness_report"):
        wr = st.session_state.wellness_report
        st.markdown(wr)
        st.download_button("📥 Download Report (TXT)", data=wr, file_name=f"Student_Wellness_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", use_container_width=True)
        if st.button("🔄 Check Again", use_container_width=True):
            st.session_state.wellness_report = ""
            st.rerun()
    else:
        st.markdown("""
        <div class="glass-card">
            <h4>Check Yourself in 2 Minutes</h4>
            <ul>
                <li>⚖️ BMI & Physical Fitness</li>
                <li>🧠 Stress, Mood & Focus Check</li>
                <li>🏠 Home Workout (No Equipment)</li>
                <li>🧘 Yoga & Breathing Exercises</li>
                <li>🚨 Doctor Alert for Warning Signs</li>
                <li>📅 7-Day Practical Action Plan</li>
            </ul>
            <div class="service-box">
                <strong>Note:</strong> For awareness only. Not medical diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# AI STUDENT DAILY ENVIRONMENTAL FOOTPRINT & POLLUTION TRACKER
# --------------------------------------------------

st.subheader("🌱 AI Student Daily Environmental Footprint & Pollution Tracker")
st.caption("Track your real daily actions from morning to night and see your true environmental impact with evidence-based calculations.")

eco_c1, eco_c2 = st.columns([1, 1], gap="large")

with eco_c1:
    st.markdown("### 🚍 Transport")
    eco_walk = st.number_input("Walking/Cycling (km)", min_value=0.0, step=0.1)
    eco_bus = st.number_input("Bus/Public Transport (km)", min_value=0.0, step=0.1)
    eco_rickshaw = st.number_input("Rickshaw/CNG/Auto (km)", min_value=0.0, step=0.1)
    eco_bike = st.number_input("Motorbike/Scooter (km)", min_value=0.0, step=0.1)
    eco_car = st.number_input("Car/Private Vehicle (km)", min_value=0.0, step=0.1)
    
    st.markdown("### 🍛 Food & Diet")
    eco_veg = st.number_input("Veg/Home Meal (meals)", min_value=0, step=1)
    eco_egg = st.number_input("Egg-based Meal (meals)", min_value=0, step=1)
    eco_chicken = st.number_input("Chicken (meals)", min_value=0, step=1)
    eco_beef = st.number_input("Beef/Mutton (meals)", min_value=0, step=1)
    eco_fastfood = st.number_input("Fast Food/Packaged Food (meals/snacks)", min_value=0, step=1)
    
    st.markdown("### 🛍️ Plastic, Packaging & Waste")
    eco_bottle = st.number_input("Plastic Bottle Used (pcs)", min_value=0, step=1)
    eco_bag = st.number_input("Polythene Bag Used (pcs)", min_value=0, step=1)
    eco_takeaway = st.number_input("Takeaway with plastic/container (times)", min_value=0, step=1)
    eco_disposable = st.number_input("Disposable cup/plate/spoon (pcs)", min_value=0, step=1)
    
    st.markdown("### 💡 Electricity & Water")
    eco_fan = st.number_input("Fan used (hours)", min_value=0.0, step=0.5)
    eco_ac = st.number_input("AC used (hours)", min_value=0.0, step=0.5)
    eco_light = st.number_input("Lights used (hours)", min_value=0.0, step=0.5)
    eco_charge = st.number_input("Phone/Laptop charging (hours)", min_value=0.0, step=0.5)
    eco_shower = st.number_input("Shower/Bath usage (times)", min_value=0, step=1)
    eco_wash = st.number_input("Clothes wash (times)", min_value=0, step=1)
    
    st.markdown("### 🗑️ Other")
    eco_paper = st.number_input("Printed/A4 paper used (sheets)", min_value=0, step=1)
    eco_online_del = st.number_input("Online food/grocery delivery (orders)", min_value=0, step=1)
    eco_tissue = st.number_input("Tissue paper used (approx rolls/sheets count)", min_value=0, step=1)
    
    eco_calc = st.button("🧮 Calculate My Daily Footprint", type="primary", use_container_width=True)
    
    if eco_calc:
        api_key = get_api_key()
        if not api_key:
            st.error("GEMINI_API_KEY missing.")
        else:
            try:
                with st.spinner("Calculating with research-based emission factors..."):
                    client = genai.Client(api_key=api_key)
                    prompt = f"""You are an environmental scientist specializing in student carbon/waste footprints.

Use realistic, paper-based emission factors (aligned with IPCC, DEFRA 2024 style, Our World in Data).

DAILY LOG:
Transport: walk/cycle {eco_walk}km, bus {eco_bus}km, rickshaw/cng {eco_rickshaw}km, bike {eco_bike}km, car {eco_car}km
Food: veg {eco_veg}, egg {eco_egg}, chicken {eco_chicken}, beef/mutton {eco_beef}, fastfood {eco_fastfood}
Plastic/Waste: bottle {eco_bottle}, bag {eco_bag}, takeaway {eco_takeaway}, disposable {eco_disposable}
Electricity/Water: fan {eco_fan}h, ac {eco_ac}h, light {eco_light}h, charge {eco_charge}h, shower {eco_shower}, wash {eco_wash}
Others: paper {eco_paper}, delivery {eco_online_del}, tissue {eco_tissue}

Calculate and show:

## DAILY ENVIRONMENTAL FOOTPRINT SUMMARY
Total CO₂e (kg CO₂ equivalent) for today

## BREAKDOWN BY CATEGORY
Transport | Food | Plastic & Packaging | Energy (Electricity) | Water/Waste | Others - with kg CO₂e + %

## TOP POLLUTER
Which action caused most impact + why

## REAL COMPARISONS
e.g. "Equals driving X km by car" / "Needs ~X trees for 1 year to offset this single day" (pragmatic)

## HIDDEN IMPACTS
Small actions often ignored

## WHAT YOU CAN DO TOMORROW (PRACTICAL)
At least 5 realistic steps for a student (low cost/no cost)

## 7-DAY ECO-CHALLENGE
Day1-Day7 with doable targets to cut footprint

## AWARENESS FOR A STUDENT
Explain in simple words how small choices add up over month/year

## EVIDENCE NOTE
State factors used are approximate but research-aligned for learning awareness

Be honest, educational, Bangladesh/Global relatable. Focus on action over guilt."""
                    res, used = generate_with_retry(client, prompt, model_name)
                    st.session_state.eco_report = res
                    st.success("Footprint calculated!")
                    st.rerun()
            except Exception as e:
                st.error(classify_api_error(e))

with eco_c2:
    st.markdown("### Your Daily Impact Report")
    if st.session_state.get("eco_report"):
        er = st.session_state.eco_report
        st.markdown(er)
        st.download_button("📥 Download Report (TXT)", data=er, file_name=f"Student_Daily_Footprint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", use_container_width=True)
        if st.button("🔄 Track Another Day", use_container_width=True):
            st.session_state.eco_report = ""
            st.rerun()
    else:
        st.markdown("""
        <div class="glass-card">
            <h4>Track From Morning To Night</h4>
            <p>See real impact of your daily choices:</p>
            <ul>
                <li>🚲 Transport footprint</li>
                <li>🍽️ Food choices impact</li>
                <li>♻️ Plastic & packaging</li>
                <li>⚡ Energy & water use</li>
                <li>📊 kg CO₂e breakdown</li>
                <li>🌳 Tree-offset comparison</li>
                <li>✅ Easy ways to reduce</li>
                <li>📆 7-Day Eco-Challenge</li>
            </ul>
            <div class="service-box">
                <strong>Goal:</strong> Awareness through numbers - not guilt, action.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

st.subheader("❓ Frequently Asked Questions")

faq_col1, faq_col2 = st.columns(2, gap="medium")

with faq_col1:
    with st.expander("What type of documents can I summarize?"):
        st.write("You can upload standard `.txt` files, PDF documents (research papers, theses, dissertations), or paste any text directly up to 60,000 characters.")
    
    with st.expander("How does the AI protect academic integrity?"):
        st.write("Our system is designed to summarize and structure your authentic content without introducing fabricated data or plagiarizing external sources.")

with faq_col2:
    with st.expander("Can I generate summaries in different languages?"):
        st.write("Yes! You can translate and summarize your documents into over 20 languages including Bengali, Spanish, French, German, and Arabic instantly.")

    with st.expander("How do backup models work?"):
        st.write("If the primary Gemini model experiences temporary high traffic or rate limits, the app automatically retries and falls back to our secondary model seamlessly.")

st.markdown(
    """
    <div style="
        text-align: center;
        color: #94a3b8;
        padding: 24px 12px 12px 12px;
        font-size: 0.9rem;
    ">
        <strong>AI Text Summarizer Pro | Academic Edition</strong><br>
        Built with Python, Streamlit, and Google Gemini API &nbsp;|&nbsp; Committed to Academic Excellence
    </div>
    """,
    unsafe_allow_html=True,
)
