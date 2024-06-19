from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_postgres.vectorstores import DistanceStrategy
from typing import List

embeddings = OllamaEmbeddings(model="snowflake-arctic-embed:335m")

file_path = "d77cf8fd-03ce-4fa5-a09c-bcb4e8d2e6c7.pdf"

loader = PyPDFLoader(file_path=file_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

data = loader.load_and_split(text_splitter=text_splitter)

cleaned_data: List[Document] = []

for doc in data:
  cleaned_data.append(Document(page_content=doc.page_content.replace('\n',' '), metadata=doc.metadata))

connection_string = "postgresql+psycopg://pgadm:pgadm1234#@p1-east.cluster-cafary0vpprp.us-east-1.rds.amazonaws.com:5432/db1"

vectorstore = PGVector(
     embeddings=embeddings,
     collection_name="my_collection",
     connection=connection_string,
     distance_strategy = DistanceStrategy.COSINE,
     use_jsonb=True,
 )

vectorstore.add_documents(cleaned_data) 
#vectorstore.add_documents(data) 
