import streamlit as st
import os

# Try importing the modern PdfReader

from PyPDF2 import PdfReader  # Modern PyPDF2 import for reading PDF files
    except ImportError:
        # Fallback to older PyPDF2 import syntax for compatibility
            from PyPDF2 import PdfFileReader as PdfReader

from langchain_openai import ChatOpenAI  # OpenAI Chat LLM integration
from langchain.text_splitter import CharacterTextSplitter  # Splits text into manageable chunks
from langchain_community.vectorstores import FAISS  # FAISS for creating and searching vector stores
from langchain_openai import OpenAIEmbeddings  # OpenAI embeddings to vectorize text
from langchain.memory import ConversationBufferMemory  # Memory to track chat history
from langchain.chains import ConversationalRetrievalChain  # Chain for conversation with retrieval
from langchain.prompts import PromptTemplate  # Template for customizing AI prompts

# Set OpenAI API key from Streamlit secrets configuration
openai_key = st.secrets.get("OPENAI_API_KEY", None)
if not openai_key:
    st.error("Missing OpenAI API key. Please set it in your secrets configuration.")
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_key  # Set the key as an environment variable

# Page configuration
st.set_page_config(page_title="Chat with PDF", page_icon="\ud83d\udcda")
st.title("Chat with your PDF \ud83d\udcda")

# Initialize session state variables to maintain chat state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7  # Default temperature value
if "feedback" not in st.session_state:
    st.session_state.feedback = []  # Store feedback

# Feedback form
def feedback_form():
    """Display feedback form after each response."""
    st.markdown("**Was this answer helpful?**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes", key=f"yes_{len(st.session_state.feedback)}"):
            st.session_state.feedback.append("Yes")
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎 No", key=f"no_{len(st.session_state.feedback)}"):
            st.session_state.feedback.append("No")
            st.warning("Sorry to hear that! We'll improve.")

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files using PyPDF2."""
    text = ""
    for pdf in pdf_docs:  # Iterate through each uploaded PDF file
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:  # Extract text page by page
            page_text = page.extract_text() if hasattr(page, 'extract_text') else ""
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    """Split the extracted text into smaller chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",  # Split text by newline
        chunk_size=500,  # Maximum size of each text chunk
        chunk_overlap=100,  # Overlap between chunks to maintain context
        length_function=len  # Measure chunk length using the `len` function
    )
    return text_splitter.split_text(text)

def get_conversation_chain(vectorstore, temperature):
    """Create a conversational retrieval chain using the vector store and OpenAI model."""
    llm = ChatOpenAI(temperature=temperature, model_name='gpt-4o')  # Initialize the OpenAI Chat model
    template = """You are a helpful AI assistant that helps users understand their PDF documents.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:"""  # Instruction template for AI responses
    
    # Create a prompt template
    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)
    
    # Set up conversation memory for managing chat history
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    # Combine the components into a conversational retrieval chain
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),  # Use vectorstore as the retriever
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

def process_docs(pdf_docs, temperature):
    """Process uploaded PDFs: extract text, split into chunks, and create embeddings."""
    try:
        raw_text = get_pdf_text(pdf_docs)  # Extract text from PDFs
        if not raw_text:  # Handle cases where no text is extracted
            st.error("No text extracted from the uploaded PDFs. Please upload valid PDF files.")
            return False
        
        text_chunks = get_text_chunks(raw_text)  # Split text into chunks
        embeddings = OpenAIEmbeddings()  # Generate embeddings for text chunks
        
        # Create a FAISS vector store to efficiently search the embeddings
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
        # Initialize the conversation chain with user-defined temperature
        st.session_state.conversation = get_conversation_chain(vectorstore, temperature)
        st.session_state.processComplete = True  # Flag processing as complete
        return True
    except Exception as e:  # Handle any errors during processing
        st.error(f"An error occurred during processing: {str(e)}")
        return False

# Sidebar for PDF upload and settings
with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here",
        type="pdf",  # Restrict file uploads to PDFs only
        accept_multiple_files=True  # Allow multiple files to be uploaded
    )
    st.subheader("Settings")
    temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7, 0.1)
    st.session_state.temperature = temperature

    if st.button("Process") and pdf_docs:  # Process files when button is clicked
        with st.spinner("Processing your PDFs..."):
            success = process_docs(pdf_docs, temperature)
            if success:
                st.success("Processing complete!")
                st.info("I’ve summarized key insights. Ask me anything about your documents!")

# Main chat interface
if st.session_state.processComplete:  # Display chat interface only if processing is complete
    user_question = st.chat_input("Ask a question about your documents:")  # User input field
    if user_question:  # Process user question
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": user_question})  # Query the conversation chain
                st.session_state.chat_history.append(("You", user_question))  # Append user question to history
                bot_answer = response.get("answer", "Sorry, I couldn't generate a response.")
                st.session_state.chat_history.append(("Bot", bot_answer))  # Append bot's response to history
        except Exception as e:  # Handle errors during response generation
            st.error(f"An error occurred during chat: {str(e)}")
        
        # Display response feedback option
        feedback_form()
    
    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
else:
    st.write("\ud83d\udc4b Upload your PDFs in the sidebar to get started!")
