import streamlit as st
try:
    from PyPDF2 import PdfReader  # Importing PdfReader to handle PDF file reading
except ImportError as e:
    st.error("PyPDF2 is not installed. Please ensure it is in the requirements.txt file and redeploy the app.")
    raise e

from langchain_openai import ChatOpenAI  # Importing OpenAI's Chat model
from langchain.text_splitter import CharacterTextSplitter  # Splits text into manageable chunks
from langchain_community.vectorstores import FAISS  # FAISS for creating a vector store
from langchain_openai import OpenAIEmbeddings  # OpenAI embeddings for text conversion
from langchain.memory import ConversationBufferMemory  # Memory to manage conversation history
from langchain.chains import ConversationalRetrievalChain  # Chain to handle retrieval-based conversation
from langchain.prompts import PromptTemplate  # Custom prompt template for conversation
import os

# Set OpenAI API key from secrets
openai_key = st.secrets.get("OPENAI_API_KEY", None)  # Fetch OpenAI API key from Streamlit secrets
if not openai_key:
    st.error("Missing OpenAI API key. Please set it in your secrets configuration.")
    st.stop()  # Stop the app if API key is not provided
os.environ["OPENAI_API_KEY"] = openai_key

# Page configuration
st.set_page_config(page_title="Chat with PDF", page_icon="\ud83d\udcda")
st.title("Chat with your PDF \ud83d\udcda")

# Initialize session state variables
if "conversation" not in st.session_state:  # Conversation chain to handle the chat
    st.session_state.conversation = None
if "chat_history" not in st.session_state:  # Chat history to display past messages
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:  # Flag to check if PDF processing is complete
    st.session_state.processComplete = None

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:  # Loop through uploaded PDFs
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:  # Extract text from each page
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    """Split text into smaller chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,  # Each chunk will have up to 1000 characters
        chunk_overlap=200,  # Overlap of 200 characters between chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversation_chain(vectorstore):
    """Create a conversation chain using the vector store and OpenAI model."""
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o')  # Initialize the OpenAI chat model
    
    # Custom prompt to guide the AI assistant
    template = """You are a helpful AI assistant that helps users understand their PDF documents.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',  # Store the chat history in session state
        return_messages=True
    )
    
    # Create a conversational retrieval chain to integrate the model and vector store
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return conversation_chain

def process_docs(pdf_docs):
    """Process uploaded PDFs to extract text, split into chunks, and create embeddings."""
    try:
        raw_text = get_pdf_text(pdf_docs)  # Extract text from PDFs
        if not raw_text:
            st.error("No text extracted from the uploaded PDFs. Please upload valid PDF files.")
            return False
        
        text_chunks = get_text_chunks(raw_text)  # Split text into chunks
        
        embeddings = OpenAIEmbeddings()  # Convert text chunks into embeddings
        
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)  # Create FAISS vector store
        
        st.session_state.conversation = get_conversation_chain(vectorstore)  # Initialize the conversation chain
        
        st.session_state.processComplete = True  # Mark processing as complete
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
        accept_multiple_files=True  # Allow multiple file uploads
    )
    
    if st.button("Process") and pdf_docs:
        with st.spinner("Processing your PDFs..."):
            success = process_docs(pdf_docs)  # Start processing the uploaded PDFs
            if success:
                st.success("Processing complete!")

# Main chat interface
if st.session_state.processComplete:  # Only display chat interface if processing is complete
    user_question = st.chat_input("Ask a question about your documents:")
    
    if user_question:
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({  # Get AI response based on user input
                    "question": user_question
                })
                st.session_state.chat_history.append(("You", user_question))  # Append user question to chat history
                st.session_state.chat_history.append(("Bot", response["answer"]))  # Append bot's answer
        except Exception as e:
            st.error(f"An error occurred during chat: {str(e)}")

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

# Display initial instructions if no PDF is processed
else:
    st.write("\ud83d\udc4b Upload your PDFs in the sidebar to get started!")
