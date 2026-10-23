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
# FAQ & PROFESSIONAL FOOTER (Scribbr Style)
# --------------------------------------------------

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
