import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

st.set_page_config(page_title="YLori AI Travel Agent", layout="wide")
st.title("YLori AI Travel Agent (Hugging Face Version)")

# Load documents from travel websites
ota_urls = [
    "https://www.expedia.com/Destinations-In-Europe.d6022967.Hotel-Destinations",
    "https://www.kayak.com/travel-guides",
    "https://www.booking.com/articles/en/10-most-popular-destinations.html",
    "https://www.agoda.com/en-gb/travel-guides"
]

@st.cache_resource(show_spinner=True)
def load_data():
    loader = WebBaseLoader(ota_urls)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(texts, embeddings)

vectorstore = load_data()

# Use a free Hugging Face text generation model
qa_chain = RetrievalQA.from_chain_type(
    llm=HuggingFaceHub(
        repo_id="google/flan-t5-base",  # or another Hugging Face model
        model_kwargs={"temperature": 0.5, "max_length": 512}
    ),
    retriever=vectorstore.as_retriever()
)

query = st.text_input("Ask me about travel destinations, hotels, or trip planning:")
if query:
    answer = qa_chain.run(query)
    st.markdown("### Answer")
    st.write(answer)
