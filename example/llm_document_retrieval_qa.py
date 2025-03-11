from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain_ollama import OllamaEmbeddings

# -------------------------------
# Initialize the Large Language Model (LLM)
# -------------------------------

# Using Google Gemini model for generating responses.
# Note: The API key should be securely stored instead of hardcoding it.
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-thinking-exp-01-21', api_key='AIzaSyAPFYvfLoDCAzvAw1zdaZBS05IIAWanXWI', temperature=0.0)

# Alternative LLMs (commented out) that can be used instead:
# llm = Ollama(model="llama3.2-vision", temperature=0.0, repeat_last_n=-1)
# llm = Ollama(model="deepseek-r1:14b", temperature=0.0, repeat_last_n=-1, num_ctx=10000)
# llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-8b', api_key='x', temperature=0.0)

# -------------------------------
# Load Documents for Retrieval
# -------------------------------

# Load all documents from the "docs" directory.
# DirectoryLoader automatically loads text-based files (e.g., .txt, .pdf, .md).
loader = DirectoryLoader("filings2")
docs = loader.load()

# -------------------------------
# Create Document Embeddings
# -------------------------------

# Generate embeddings for the loaded documents using the HuggingFace "all-MiniLM-L6-v2" model.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------------------
# Store Documents in a FAISS Vector Database
# -------------------------------

# Convert the documents into a vectorized format using FAISS for efficient similarity search.
vectorstore = FAISS.from_documents(docs, embeddings)

# Set up a retriever that will fetch the most relevant document(s) based on a query.
# The retriever is set to return only the top 1 most relevant result (k=1).
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# -------------------------------
# Setup Retrieval-Based QA Chain
# -------------------------------

# Create a RetrievalQA chain that uses the selected LLM and the retriever.
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# -------------------------------
# Define the Query and Run QA
# -------------------------------

# Define a question prompt for the QA system.
# The LLM is instructed to act as a security expert, ensuring honesty and truthfulness in its responses.
query = 'You are a security expert who is answering questions from clients. ' \
        'All answers must be truthful and very honest. ' \
        'If you dont find the answer, you respond by saying, "I dont have info on that". ' \
        'Where does Microsoft operate its datacenters according to the report? Use exact verbiage quoted and let me know what section it is from'

# Execute the retrieval-based QA system.
result = qa_chain.run(query)

# Print the response from the LLM.
print(result)
