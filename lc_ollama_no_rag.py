from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough


embeddings = OllamaEmbeddings(model="nomic-embed-text")

model = Ollama(model="llama3")

rag_template = """Answer the Question: {question}"""

rag_prompt = ChatPromptTemplate.from_template(rag_template);

rag_chain = (
        rag_prompt
        | model
        | StrOutputParser() 
        )

#print(rag_chain.invoke("Tell me about Snowflake Zero Copy Clone"))

print(rag_chain.invoke("Tell about QuantumScape Financial statements"))
