import sys
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_postgres.vectorstores import DistanceStrategy


embeddings = OllamaEmbeddings(model="nomic-embed-text")

model = Ollama(model="llama3",num_gpu=4)

connection_string = "postgresql+psycopg://pgadm:pgadm1234#@p1-east.cluster-cafary0vpprp.us-east-1.rds.amazonaws.com:5432/db1"

vectorstore = PGVector(
     embeddings=embeddings,
     collection_name="my_collection",
     connection=connection_string,
     distance_strategy = DistanceStrategy.COSINE,
     use_jsonb=True,
 )

print(vectorstore.similarity_search(sys.argv[1], k=10))

retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

rag_template = """Answer the question only based on the context supplied part of this request. Dont try to make up answer on your own.

{context} 

Question: {question}
"""

rag_prompt = PromptTemplate(template=rag_template,input_variables=["context","question"]);

rag_chain = (
        { "context": retriever , "question": RunnablePassthrough() }
        | rag_prompt
        | model
        | StrOutputParser() 
        )

print(rag_chain.invoke(sys.argv[1]))
