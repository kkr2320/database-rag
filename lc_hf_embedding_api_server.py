import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough , RunnableLambda 
from typing import List
from fastapi import FastAPI, Request
from langserve import add_routes

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#model_name = "nomic-ai/nomic-embed-text-v1"
model_name = "Snowflake/snowflake-arctic-embed-l"
model_kwargs = {
    'device': 'cpu',
    'trust_remote_code':True
    }
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def embed_text(args: str) -> List[float]:
    return embeddings.embed_query(args)

embed_chain = RunnableLambda(embed_text) 
            

#Define the LangServe with FastAPI
app = FastAPI(
    title="Discover LangChain Embedding API Server",
    version="1.0",
    description="An api server using Langchain's Runnable interfaces",
)

#Add the API Path to call the Langchain Executor framework
add_routes(
    app,
    embed_chain,
    path="/dfsai/embedtext",)

# Run the Model as API Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app , host="localhost", port=8002)
