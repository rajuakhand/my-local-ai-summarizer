
import io
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Optional file support
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# Page configuration
st.set_page_config(
    page_title="Local AI Text Summarizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Supported languages
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

SUMMARY_LENGTHS = {
    "Short": "Write a concise summary in approximately 3–5 sentences.",
    "Medium": "Write a clear summary in approximately 6–10 sentences.",
    "Detailed": "Write a detailed summary with the main ideas, important facts, and relevant context.",
}


# Custom styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #101827 0%, #172554 55%, #312e81 100%);
        color: #f8fafc;
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        text-align: center;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.14);
        padding: 18px;
        border-radius: 15px;
        margin-bottom: 12px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        min-height: 43px;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.07);
        padding: 12px;
        border-radius: 12px;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "last_language" not in st.session_state:
    st.session_state.last_language = "English"


# Header
st.markdown(
    '<div class="main-title">🧠 Local AI Text Summarizer</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Summarize text with AI in 20 languages.</div>',
    unsafe_allow_html=True,
)


# Sidebar settings
with st.sidebar:
    st.title("⚙️ Settings")

    selected_language = st.selectbox(
        "Output language",
        list(LANGUAGES.keys()),
        index=0,
    )

    summary_length = st.selectbox(
        "Summary length",
        ["Short", "Medium", "Detailed"],
        index=1,
    )

    model_name = st.text_input(
        "Ollama model",
        value="llama3.2:1b",
        help="The model must be installed on the Ollama server.",
    )

    st.divider()
    st.subheader("✨ Features")
    st.markdown(
        """
        - 20 output languages
        - Adjustable summary length
        - TXT and PDF upload
        - Download summary as TXT
        - Right-to-left language support
        - Local Ollama model integration
        """
    )

    st.caption("Powered by Streamlit, LangChain and Ollama")


# Main layout
left_column, right_column = st.columns([1.2, 0.8], gap="large")

with left_column:
    st.subheader("📝 Your Text")

    uploaded_file = st.file_uploader(
        "Upload a TXT or PDF file",
        type=["txt", "pdf"],
        help="You can upload a text file or a PDF document.",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".txt"):
                uploaded_text = uploaded_file.getvalue().decode(
                    "utf-8",
                    errors="replace",
                )
            else:
                if PdfReader is None:
                    st.error(
                        "PDF support is unavailable. Add pypdf to requirements.txt."
                    )
                    uploaded_text = ""
                else:
                    pdf_reader = PdfReader(
                        io.BytesIO(uploaded_file.getvalue())
                    )
                    uploaded_text = "\n".join(
                        page.extract_text() or ""
                        for page in pdf_reader.pages
                    )

            if uploaded_text:
                st.session_state.input_text = uploaded_text

        except Exception as error:
            st.error(f"Could not read the uploaded file: {error}")

    input_text = st.text_area(
        "Paste or edit your text here",
        key="input_text",
        height=300,
        placeholder="Paste your article, notes, essay, or document here...",
    )

    word_count = len(input_text.split())
    character_count = len(input_text)

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric("Words", f"{word_count:,}")

    with metric2:
        st.metric("Characters", f"{character_count:,}")

    generate_column, clear_column = st.columns(2)

    with generate_column:
        generate_clicked = st.button(
            "✨ Generate Summary",
            type="primary",
            use_container_width=True,
        )

    with clear_column:
        clear_clicked = st.button(
            "🗑️ Clear All",
            use_container_width=True,
        )

    if clear_clicked:
        st.session_state.input_text = ""
        st.session_state.summary = ""
        st.session_state.last_language = "English"
        st.rerun()


# Generate summary
if generate_clicked:
    if not input_text.strip():
        st.warning("Please enter some text or upload a document first.")

    elif not model_name.strip():
        st.error("Please enter a valid Ollama model name.")

    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a professional multilingual text summarization assistant.

                    Summarize the user's text accurately in {language}.
                    Follow these instructions:
                    - {length_instruction}
                    - Preserve the original meaning and important facts.
                    - Do not invent information.
                    - Use clear, natural language.
                    - Return only the summary.
                    - Do not add an introduction such as "Here is the summary".
                    """,
                ),
                (
                    "human",
                    "Summarize the following text:\n\n{text}",
                ),
            ]
        )

        try:
            with st.spinner("🤖 AI is generating your summary..."):
                llm = ChatOllama(
                    model=model_name.strip(),
                    temperature=0.2,
                )

                chain = prompt | llm

                result = chain.invoke(
                    {
                        "language": LANGUAGES[selected_language],
                        "length_instruction": SUMMARY_LENGTHS[summary_length],
                        "text": input_text,
                    }
                )

                st.session_state.summary = result.content
                st.session_state.last_language = selected_language

        except Exception as error:
            st.error(
                "Could not generate the summary. Check that Ollama is running, "
                "the selected model is installed, and the server is reachable."
            )
            with st.expander("Technical error details"):
                st.code(str(error))


# Summary output
with right_column:
    st.subheader("📄 Your Summary")

    if st.session_state.summary:
        summary_text = st.session_state.summary

        # Right-to-left display for Arabic, Urdu and Persian
        rtl_languages = {"Arabic", "Urdu", "Persian"}
        if st.session_state.last_language in rtl_languages:
            st.markdown(
                f"""
                <div dir="rtl" style="
                    background: rgba(255,255,255,0.08);
                    border: 1px solid rgba(255,255,255,0.15);
                    border-radius: 14px;
                    padding: 20px;
                    line-height: 1.9;
                    white-space: pre-wrap;
                ">{summary_text}</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="
                    background: rgba(255,255,255,0.08);
                    border: 1px solid rgba(255,255,255,0.15);
                    border-radius: 14px;
                    padding: 20px;
                    line-height: 1.8;
                ">
                """,
                unsafe_allow_html=True,
            )
            st.write(summary_text)
            st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            label="⬇️ Download Summary (.txt)",
            data=summary_text,
            file_name="ai_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.caption("Copy the summary using the text area below.")

        st.text_area(
            "Copyable summary",
            value=summary_text,
            height=180,
            help="Select the text and copy it. Direct clipboard access is not enabled here.",
        )

        if st.button("🔄 Start New Summary", use_container_width=True):
            st.session_state.summary = ""
            st.rerun()

    else:
        st.info(
            "Your summary will appear here after you click Generate Summary."
        )

        st.markdown(
            """
            <div class="feature-card">
                <h4>🚀 How to use</h4>
                <p>1. Enter text or upload a file.</p>
                <p>2. Select your output language.</p>
                <p>3. Choose the summary length.</p>
                <p>4. Click Generate Summary.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Footer
st.divider()
st.markdown(
    "<center>Built with ❤️ using Python, Streamlit, LangChain and Ollama</center>",
    unsafe_allow_html=True,
)

