import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="Local AI Summarizer",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #182b3f);
    color: #f8fafc;
}
h1, h2, h3, p, label {
    color: #f8fafc !important;
}
.hero {
    text-align: center;
    padding: 24px 10px;
}
.hero p {
    color: #bdc9d8 !important;
    font-size: 17px;
}
.stTextArea textarea {
    background: #f8fafc;
    color: #111827;
    border-radius: 12px;
}
.stButton button {
    border-radius: 10px;
    min-height: 45px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🧠 Local AI Summarizer</h1>
    <p>Turn long text into clear, short summaries in seconds.</p>
</div>
""", unsafe_allow_html=True)

language = st.selectbox(
    "🌐 Summary language",
    ["English", "বাংলা"]
)

text = st.text_area(
    "📝 Paste your text",
    height=240,
    placeholder="Paste an article, notes, or any long text here..."
)

st.caption(f"Character count: {len(text):,}")

col1, col2 = st.columns(2)

with col1:
    summarize = st.button(
        "✨ Summarize",
        type="primary",
        use_container_width=True
    )

with col2:
    clear = st.button(
        "🧹 Clear text",
        use_container_width=True
    )

if clear:
    st.session_state["input_text"] = ""
    st.rerun()

if summarize:
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        try:
            with st.spinner("🤖 AI is preparing your summary..."):
                llm = ChatOllama(
                    model="llama3.2:1b",
                    temperature=0
                )

                prompt = ChatPromptTemplate.from_template(
                    "Summarize the text below in exactly 3 short sentences. "
                    "Write the summary in {language}. "
                    "Keep the important information and do not add facts.\n\n"
                    "Text:\n{text}"
                )

                chain = prompt | llm
                result = chain.invoke({
                    "text": text,
                    "language": language
                })

            st.session_state["summary"] = result.content

        except Exception as error:
            st.error(f"Something went wrong: {error}")
            st.info(
                "Check that Ollama is running and llama3.2:1b is installed."
            )

if st.session_state.get("summary"):
    st.divider()
    st.subheader("📄 Your Summary")
    st.write(st.session_state["summary"])

    st.code(st.session_state["summary"], language=None)
    st.caption("Copy the summary using the copy icon on the text box above.")

st.divider()
st.caption("🔒 Runs locally with Ollama • No paid AI API key required")
