import os , sys , uuid , boto3 , psycopg  , json
import numpy as np
from langchain_community.document_loaders import S3FileLoader
from urllib.parse import urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pgvector.psycopg import register_vector
from langchain_community.document_loaders import PyPDFLoader
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

file_path = sys.argv[3]

if file_path.startswith("s3:") :
   o = urlparse(file_path, allow_fragments=False)
   s3 = boto3.client("s3")
   bucket = o.netloc 
   key = o.path
   key = key.replace('/','',1)
   print("Bucket and File " + bucket + " " + key)
   myuuid = str(uuid.uuid4())
   s3.download_file(bucket, key, "/tmp/" + myuuid)
   loader = PyPDFLoader(file_path="/tmp/" +  myuuid)
   data = loader.load()
   os.remove("/tmp/" + myuuid)
else :
   loader = PyPDFLoader(file_path=file_path)
   data = loader.load()

cleaned_data: List[Document] = []
for doc in data:
  cleaned_data.append(Document(page_content=doc.page_content.replace('\n',' '), metadata=doc.metadata))

print(cleaned_data[0].metadata)

connection_string = "postgresql://pgadm:pgadm1234#@p1-east.cluster-cafary0vpprp.us-east-1.rds.amazonaws.com:5432/db1"


with psycopg.connect(connection_string) as conn :
  register_vector(conn)
  print("test")
  for doc in cleaned_data : 
    ## Generate Embeedings for each page 
    embedding = embeddings.embed_query (doc.page_content) 
    embed_vector = np.array(embedding)
    rslt = conn.execute("INSERT INTO dfs_financial_documents ( doc_type , doc_date , doc_page_content , embeddings , additional_metadata ) VALUES ( %s , %s , %s , %s , %s ) " , 
                                                       ( sys.argv[1], sys.argv[2] , doc.page_content , embed_vector , json.dumps(doc.metadata) ) )

     
 
