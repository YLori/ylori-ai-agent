import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

st.set_page_config(page_title="YLori Travel Agent", layout="wide")
st.title("YLori AI Travel Agent")

# Optional: You can still set an OpenAI key if you want to use ChatOpenAI later
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.warning("OPENAI_API_KEY is not set — you can only use HuggingFaceEmbeddings right now.")

# Top OTA sites for travel info
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

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
    retriever=vectorstore.as_retriever()
)

query = st.text_input("Ask me about travel destinations, hotels, or trip planning:")
if query:
    answer = qa_chain.run(query)
    st.markdown("### Answer")
    st.write(answer)
