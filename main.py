from langchain.embeddings import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Step 1: Store Embeddings in AstraDB Vector Store (Fresh Data)
def store_vectors():
    # Load the PDF Document
    pdf_loader = PyPDFLoader("./sources/Canada.pdf")  
    documents = pdf_loader.load()
    
    # Split the Text into Chunks for Better Retrieval
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Convert Text to Vector Embeddings using Hugging Face
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Initialize AstraDB and clear previous data
    vectorstore = AstraDBVectorStore(
        collection_name="canada",
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_NAMESPACE,
    )    
    
    vectorstore.delete_collection()
    
    # Store new vectors
    vectorstore.add_documents(documents=docs)

    print("Vectors stored successfully.")

# Step 2: Fetch Embeddings and Set Up RAG Chain
def fetch_and_query():
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    vectorstore = AstraDBVectorStore(
        collection_name="canada",
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_NAMESPACE,
    )    

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 1, "score_threshold": 0.5},
    )

    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0})
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    
    # Ask a Question and Get a Response
    query = "What does the document say about World War II?"
    response = qa_chain.run(query)
    
    print("AI Response:", response)
