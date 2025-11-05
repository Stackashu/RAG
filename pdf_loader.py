### RAG Pipeline - Data Ingestion to vector DB Pipeline
import os
from pathlib import Path 
from langchain_community.document_loaders import PyPDFLoader , PyMuPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import sentence_transformers
from sympy import N
from torch._dynamo.resume_execution import _try_except_tf_mode_template

from docs import document

###Read all the pdf's inside the directory

def process_all_pdf(pdf_directory):
    """Process all Pdf files in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} Pdf files to process")
    
    for pdfs in pdf_files:
        print(f"\nProcessing: {pdf_dir.name}")
        try:
            loader = PyPDFLoader(str(pdfs))
            documents = loader.load()
            
            
            for doc in documents:
                doc.metadata['source_file'] = pdfs.name
                doc.metadata['file_type'] = 'pdf'
                
            all_documents.extend(documents)
            print(f"    Loaded {len(documents)} pages")
            
        except Exception as e:
            print(f"    X Error: {e}")
    
    print(f"\nTotal Documents loaded: {len(all_documents)}")
    return all_documents

all_pdf_documents = process_all_pdf("./data/pdf")

print("pdf",all_pdf_documents)


### text splitting get into chuncks

def split_documents(documents,chunk_size = 1000 ,chunk_overlap = 200 ):
    """Split docuemnts into smaller chunks for beter RAG performnace"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len , 
        separators= ["\n\n" , "\n"," " , ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    #show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}....")
        print(f"Metadata: {split_docs[0].metadata}")
        
    return split_docs

chunks = split_documents(all_pdf_documents)
# print(chunks)



### Now here we are doing embedding and VectorDB
import numpy as np
import uuid
from typing import List , Dict , Any , Tuple 
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class EmbeddingManager():
    """Handles documnet embedding generation using SentenceTransformers"""
    
    def __init__(self , model_name:str="all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager 
        
        Args:
            mode_name : Hugging Face model name for sentence embeddings
        """
        self.model_name = model_name 
        self.model = None
        self._load_model()  ## it will load the model above
        
    def _load_model(self):
        """Load the SenteneTransformer model"""
        
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded Successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}:{e}")
            raise
        
    def generate_embeddings(self,texts: List[str]) ->np.ndarray:
        """
        Generate embeddings for a list of texts 
        
        Args:
            texts: List of text string to embed
            
        Returs:
             numpy array of embdedding with shape (len(texts)) embedding_dim"""
             
        if not self.model :
            raise ValueError("Model not found")
        print(f"Generatting embeddings for {len(texts)} texts....")
        embeddings = self.model.encode(texts,show_progress_bar=True)
        print(f"Generated embeddings with shape : {embeddings.shape}")
        return embeddings
    
    


### initialize the embedding manager 

embedding_manager = EmbeddingManager()
# print(embedding_manager)


### VectorStore

class VectoreStore():
    """Manages document embeddings in a chromaDb vector store"""
    
    def __init__(self,collection_name: str = "pdf_documents" , persist_directory: str = "./data/vector_store"):
     """
     Initialize the vector store
     
     Args:
         Collection_name : Name of the ChromaDB Collection
         persist_directory : Directory to persist the vector store
     """
     
     self.collection_name = collection_name
     self.persist_directory = persist_directory
     self.client = None 
     self.collection = None 
     self._initialize_store()
     
    
    def _initialize_store(self):
        """Inititalize ChromaDB client and collection"""
        try:
            #create persistent ChromaDB client which will have the access/reference for the chromadb 
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            #Get or Create Collection
            self.collection  = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata = {"description":"PDF documnet embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection : {self.collection.count()}")
        except Exception as e:
            print(f"Error initializeing vector store: {e}")
            raise
    
    def add_documents(self,documents: List[Any] , embeddings : np.ndarray):
        """
        Add documents and their embdeddings to the vecror store 
        
        Args:
            documnets: List of Langchain Documents
            embedding : Corresponding embedding for the docs
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of docs must match number of embeddings")
        
        print(f"Adding {len(documents)} docs to vector store...")
        
        #Prepare data for chromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i , (doc,embedding) in enumerate(zip(documents,embeddings)):
            #Generate unique Id 
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            #Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            #document content
            documents_text.append(doc.page_content)
            
            #Embeddings
            embeddings_list.append(embedding.tolist())
    
         #Add To collection
        try:
             self.collection.add(
                 ids = ids,
                 embeddings = embeddings_list,
                 metadatas = metadatas,
                 documents=documents_text
             )
             print(f"Successfully added {len(documents)} documents to vectore store")
             print(f"Total documents in collection: {self.collection.count()}")
             
        except Exception as e :
             print(f"Error adding documents to vector store: {e}")
             raise
         
         
vector_store = VectoreStore()
# print(vectorStore)

print("CHUNKS",chunks)

### Conver the text to embeddings
text = [doc.page_content for doc in chunks]

## Generate the Embeddings 

embeddings = embedding_manager.generate_embeddings(text)

##Store in the vectore database
vectorDB = vector_store.add_documents(chunks,embeddings)