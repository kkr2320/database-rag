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

model = Ollama(model="llama3", temperature=0.4, top_k=20, top_p = 0.5)

connection_string = "postgresql+psycopg://pgadm:pgadm1234#@p1-east.cluster-cafary0vpprp.us-east-1.rds.amazonaws.com:5432/db1"

vectorstore = PGVector(
     embeddings=embeddings,
     collection_name="my_collection",
     connection=connection_string,
      distance_strategy = DistanceStrategy.COSINE,
     use_jsonb=True,
 )

ctx = vectorstore.similarity_search("QuantumScape 2023 Financial statements", k=20)

rag_template = """Summarize this n detail: {question}"""

rag_prompt = PromptTemplate(template=rag_template,input_variables=["question"]);

rag_chain = (
        rag_prompt
        | model
        | StrOutputParser()
        )
print(rag_chain.invoke(ctx))

