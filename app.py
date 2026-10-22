
import html
import io
import streamlit as st
from google import genai

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


st.set_page_config(
    page_title="Local AI Text Summarizer",
    page_icon="🧠",
    layout="wide",
)

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

LENGTHS = {
    "Short": "Summarize in 3-5 concise sentences.",
    "Medium": "Summarize in 6-10 clear sentences.",
    "Detailed": (
        "Write a detailed summary with the main ideas, "
        "important facts, and relevant context."
    ),
}

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg, #101827, #172554, #312e81
        );
    }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
    }
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        margin-bottom: 25px;
    }
    .stButton button, .stDownloadButton button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "summary_language" not in st.session_state:
    st.session_state.summary_language = "English"


st.markdown(
    '<div class="main-title">🧠 AI Text Summarizer</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Summarize text in 20 languages with Gemini AI'
    '</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")

    language = st.selectbox(
        "Output language",
        list(LANGUAGES.keys()),
    )

    length = st.selectbox(
        "Summary length",
        ["Short", "Medium", "Detailed"],
        index=1,
    )

    st.divider()
    st.caption("Powered by Google Gemini API")
    st.caption("Free-tier limits may apply.")


st.subheader("📝 Input Text")

uploaded_file = st.file_uploader(
    "Upload a TXT or PDF file",
    type=["txt", "pdf"],
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".txt"):
            file_text = uploaded_file.getvalue().decode(
                "utf-8", errors="replace"
            )
        else:
            if PdfReader is None:
                st.error(
                    "PDF support is unavailable. "
                    "Add pypdf to requirements.txt."
                )
                file_text = ""
            else:
                pdf = PdfReader(
                    io.BytesIO(uploaded_file.getvalue())
                )
                file_text = "\n".join(
                    page.extract_text() or ""
                    for page in pdf.pages
                )

        if file_text:
            st.session_state.input_text = file_text

    except Exception as exc:
        st.error(f"Could not read the file: {exc}")


input_text = st.text_area(
    "Paste your text here",
    key="input_text",
    height=260,
    placeholder="Paste an article, notes, or document...",
)

word_count = len(input_text.split())
col1, col2 = st.columns(2)

col1.metric("Words", f"{word_count:,}")
col2.metric("Characters", f"{len(input_text):,}")

generate_col, clear_col = st.columns(2)

with generate_col:
    generate = st.button(
        "✨ Generate Summary",
        type="primary",
    )

with clear_col:
    clear = st.button("🗑️ Clear All")

if clear:
    st.session_state.input_text = ""
    st.session_state.summary = ""
    st.session_state.summary_language = "English"
    st.rerun()


if generate:
    if not input_text.strip():
        st.warning("Please enter text or upload a file.")
    elif len(input_text) > 100000:
        st.warning(
            "Please shorten the text to under 100,000 characters."
        )
    else:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]

            if not api_key:
                st.error(
                    "GEMINI_API_KEY is empty. "
                    "Add it in Streamlit Settings > Secrets."
                )
                st.stop()

            client = genai.Client(api_key=api_key)

            prompt = f"""
You are a professional multilingual summarization assistant.

Summarize the supplied text in {LANGUAGES[language]}.
Instructions:
- {LENGTHS[length]}
- Preserve the important facts and original meaning.
- Do not invent details.
- Use natural, clear language.
- Return only the summary.

TEXT TO SUMMARIZE:
{input_text}
"""

            with st.spinner("🤖 Gemini is summarizing your text..."):
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt,
                )

            result = response.text

            if not result or not result.strip():
                st.error(
                    "The AI returned an empty response. Try again."
                )
            else:
                st.session_state.summary = result.strip()
                st.session_state.summary_language = language

        except KeyError:
            st.error(
                "API key is missing. Open Streamlit Settings > Secrets "
                "and add GEMINI_API_KEY."
            )
        except Exception as exc:
            st.error("Summary generation failed.")
            st.caption(
                "Check your API key, model access, internet connection, "
                "and free-tier quota."
            )
            with st.expander("Technical error details"):
                st.code(str(exc))


if st.session_state.summary:
    st.divider()
    st.subheader("📄 Generated Summary")

    summary = st.session_state.summary
    rtl_languages = {"Arabic", "Urdu", "Persian"}

    if st.session_state.summary_language in rtl_languages:
        safe_summary = html.escape(summary)
        st.markdown(
            f'<div dir="rtl" style="white-space:pre-wrap;'
            f'padding:16px;background:rgba(255,255,255,0.08);'
            f'border-radius:12px;line-height:1.9;">'
            f'{safe_summary}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.write(summary)

    st.download_button(
        "⬇️ Download Summary (.txt)",
        data=summary,
        file_name="ai_summary.txt",
        mime="text/plain",
    )

    st.text_area(
        "Copy your summary",
        value=summary,
        height=180,
    )

st.divider()
st.caption(
    "Built with Python, Streamlit and Google Gemini."
)
