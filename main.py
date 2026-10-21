from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3.2:1b", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "Summarize the following text in 3 short sentences:\n\n{text}"
)

chain = prompt | llm

def summarize_text(text):
    result = chain.invoke({"text": text})
    return result.content

if __name__ == "__main__":
    print("Local AI Text Summarizer")
    print("Type 'exit' to quit.\n")

    while True:
        text = input("Enter text to summarize: ")

        if text.strip().lower() == "exit":
            print("Goodbye!")
            break

        if not text.strip():
            print("Please enter some text.")
            continue

        try:
            print("\nSummary:\n")
            print(summarize_text(text))
            print()
        except Exception as error:
            print(f"Error: {error}")
