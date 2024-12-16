import streamlit as st

try:
    # Try importing the modern PdfReader
    from PyPDF2 import PdfReader
except ImportError:
    # Fallback to older PyPDF2 import syntax
    from PyPDF2 import PdfFileReader as PdfReader

from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os

# Set OpenAI API key from secrets
openai_key = st.secrets.get("OPENAI_API_KEY", None)
if not openai_key:
    st.error("Missing OpenAI API key. Please set it in your secrets configuration.")
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_key

# Page configuration
st.set_page_config(page_title="Chat with PDF", page_icon="📚")
st.title("Chat with your PDF 📚")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() if hasattr(page, 'extract_text') else ""
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    """Split text into smaller chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_conversation_chain(vectorstore):
    """Create a conversation chain using the vector store and OpenAI model."""
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o')
    template = """You are a helpful AI assistant that helps users understand their PDF documents.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:"""
    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

def process_docs(pdf_docs):
    """Process uploaded PDFs to extract text, split into chunks, and create embeddings."""
    try:
        raw_text = get_pdf_text(pdf_docs)
        if not raw_text:
            st.error("No text extracted from the uploaded PDFs. Please upload valid PDF files.")
            return False
        text_chunks = get_text_chunks(raw_text)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.processComplete = True
        return True
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        return False

# Sidebar for PDF upload
with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here",
        type="pdf",
        accept_multiple_files=True
    )
    if st.button("Process") and pdf_docs:
        with st.spinner("Processing your PDFs..."):
            success = process_docs(pdf_docs)
            if success:
                st.success("Processing complete!")

# Main chat interface
if st.session_state.processComplete:
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": user_question})
                st.session_state.chat_history.append(("You", user_question))
                bot_answer = response.get("answer", "Sorry, I couldn't generate a response.")
                st.session_state.chat_history.append(("Bot", bot_answer))
        except Exception as e:
            st.error(f"An error occurred during chat: {str(e)}")
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
else:
    st.write("👈 Upload your PDFs in the sidebar to get started!")
