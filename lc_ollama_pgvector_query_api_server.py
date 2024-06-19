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
from fastapi import FastAPI
from langserve import add_routes
from langchain.globals import set_debug
from langchain_core.runnables import RunnableLambda , Runnable


os.set_environment["OLLAMA_MAX_LOADED_MODELS"]=4
os.set_environment["OLLAMA_NUM_PARALLEL"]=10

set_debug(True)


embeddings = OllamaEmbeddings(model="snowflake-arctic-embed:335m")

model = Ollama(model="llama3:latest" , num_gpu=4 )

connection_string = "postgresql+psycopg://pgadm:pgadm1234#@p1-east.cluster-cafary0vpprp.us-east-1.rds.amazonaws.com:5432/db1"

def get_input_args_question(input: str) -> str :
    print(str)
    return input.get('question')

vectorstore = PGVector(
     embeddings=embeddings,
     collection_name="my_collection",
     connection=connection_string,
     distance_strategy = DistanceStrategy.COSINE,
     use_jsonb=True,
 )

#print(vectorstore.similarity_search(sys.argv[1], k=10))

retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

rag_template = """
     <|begin_of_text|><|start_header_id|>system<|end_header_id|>
     You are a Financial Application to answer financial related question from the context provided. Its important you only answer the question only based on the context supplied part of this request. Dont try to make up answer on your own. If the answer can not be generated , then respond back as "Non-Determittent". <|eot_id|>

      <|start_header_id|>user<|end_header_id|>
      Context: {context}

      Question: {question}
      <|eot_id|>

      <|start_header_id>assistant<|end_header_id|>
      """

rag_prompt = PromptTemplate(template=rag_template,input_variables=["context","question"])

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)


rag_chain = (
        { "context": retriever , "question": RunnablePassthrough() } 
        | rag_prompt
        | model
        | StrOutputParser() 
        )

add_routes(
    app,
    rag_chain ,
    path="/ragQuery",)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
