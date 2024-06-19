import os
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_community.embeddings  import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from typing import List

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#model_name = "nomic-ai/nomic-embed-text-v1"
model_name = "Snowflake/snowflake-arctic-embed-l"
model_kwargs = {
    'device': 'cpu',
    'trust_remote_code':True
    }
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction = "search_query:",
    embed_instruction = "search_document:"
)

file_path = "60f59d57-efc2-4bf8-b309-564a6eb9da4f.pdf"

loader = PyPDFLoader(file_path=file_path)

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#data = loader.load_and_split(text_splitter=text_splitter)

data = loader.load()

cleaned_data: List[Document] = []
for doc in data:
  cleaned_data.append(Document(page_content=doc.page_content.replace('\n',' '), metadata=doc.metadata))

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

