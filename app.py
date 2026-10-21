import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Text Summarizer")
st.write("Paste your text below and let AI summarize it in 3 short sentences.")

text = st.text_area(
    "Enter your text",
    height=250,
    placeholder="Paste an article, paragraph, or notes here..."
)

if st.button("✨ Summarize", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            with st.spinner("AI is summarizing your text..."):
                llm = ChatOllama(
                    model="llama3.2:1b",
                    temperature=0
                )

                prompt = ChatPromptTemplate.from_template(
                    "Summarize the following text in 3 short sentences:\n\n{text}"
                )

                chain = prompt | llm
                result = chain.invoke({"text": text})

            st.subheader("📝 Summary")
            st.write(result.content)

        except Exception as error:
            st.error(f"Something went wrong: {error}")
            st.info("Make sure Ollama is running and the llama3.2:1b model is installed.")

st.divider()
st.caption("🔒 Powered by a local AI model using Ollama.")
