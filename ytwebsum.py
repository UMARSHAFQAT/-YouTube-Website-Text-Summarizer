import streamlit as st
import validators

from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain

# --- Streamlit UI Setup ---
st.set_page_config(page_title="YT/Web Summarizer", page_icon="🧠")
st.title("📽️ YouTube & 🌐 Website Text Summarizer")

st.markdown("### 🔑 Enter Your Groq API Key")
groq_api_key = st.text_input("Groq API Key", type="password")

option = st.radio("Choose Input Type", ["YouTube URL", "Website URL"])
user_input = st.text_input("Enter URL below:")
summarize_button = st.button("Summarize")

# --- Load Documents Based on Source ---
def load_documents(url, source_type):
    try:
        if source_type == "YouTube":
            loader = YoutubeLoader.from_youtube_url(url)
        else:
            loader = UnstructuredURLLoader(urls=[url])
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading content: {e}")
        return None

# --- Summarization Logic ---
if summarize_button:
    if not groq_api_key:
        st.error("❌ Please enter a valid Groq API Key.")
    elif not validators.url(user_input):
        st.error("❌ Please enter a valid URL.")
    else:
        with st.spinner("⏳ Fetching and summarizing content..."):
            source_type = "YouTube" if option == "YouTube URL" else "Website"
            documents = load_documents(user_input, source_type)

            if not documents:
                st.error("⚠️ Failed to load content.")
            else:
                total_words = sum(len(doc.page_content.split()) for doc in documents)

                try:
                    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

                    if total_words < 1500:
                        chain = load_summarize_chain(llm, chain_type="stuff")
                        st.info(f"Using STUFF method ({total_words} words)")
                    else:
                        chain = load_summarize_chain(llm, chain_type="map_reduce")
                        st.info(f"Using MAP-REDUCE method ({total_words} words)")

                    summary = chain.run(documents)
                    st.success("✅ Summary:")
                    st.write(summary)

                except Exception as e:
                    st.error(f"❌ Error while summarizing: {e}")
