import os
import sys
import numpy as np 
import psycopg
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from pgvector.psycopg import register_vector
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_community.embeddings  import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig , pipeline
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List
import torch 


#Define few Environemnt Variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

embed_model_name = sys.argv[1]
llm_model_name = sys.argv[2]

# Define Embedding Models for the Application 
#model_name = "nomic-ai/nomic-embed-text-v1"
#model_name = "Snowflake/snowflake-arctic-embed-l"
model_kwargs = {
    'device': "cpu",
    'trust_remote_code':True
    }
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=embed_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction = "search_query:",)


# Define the llm model output parser 
class output_object(BaseModel):
    Answer: str = Field(description="Answer")

out_parser = PydanticOutputParser(pydantic_object=output_object)


# Define Custom DB Vector Retreiver 
class vectordb_fetch(BaseRetriever):
   def _get_relevant_documents( 
     self, question: str, *, run_manager: CallbackManagerForRetrieverRun ) -> List[str]:
     question_array = embeddings.embed_query(question) 

     question_vectors = np.array(question_array)

     connection_string = "postgresql://pgadm:pgadm1234#@p1-east.cluster-cafary0vpprp.us-east-1.rds.amazonaws.com:5432/db1"

     with psycopg.connect(connection_string) as conn: 
       register_vector(conn)
       rslt = conn.execute("SELECT 'date reported:' || doc_date || ' ' || doc_page_content FROM dfs_financial_documents ORDER BY embeddings <=> %s LIMIT 4" , (question_vectors,) )
       Doc = rslt.fetchall()

     return Doc

# Define retriever using Custom retriever 
retriever = vectordb_fetch()

# Prompt Engineering for Models

## Llama Rag Prompt Structure
llama_rag_prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
You are a Financial Application to answer financial related question from the context provided. Its important you only answer the question only based on the context supplied part of this request. Dont try to make up answer on your own.<|eot_id|>

<|start_header_id|>user<|end_header_id|> 
Context: {context}

Question: {question}
<|eot_id|>

<|start_header_id>assistant<|end_header_id|>
"""

## Mistral Rag Prompt Structure
mistral_rag_prompt = """
<s> [INST] You are a Financial Application to answer financial related question from the context provided. Its important you only answer the question only based on the context supplied part of this request. Dont try to make up answer on your own. 

Context : {context}

Question: {question}
[/INST]

Assistant: </s>
"""

# Generate Prompt for Model Input 
if llm_model_name.startswith("meta") :
   rag_prompt_template = PromptTemplate(template=llama_rag_prompt,input_variables=["context","question"], )
elif llm_model_name.startswith("mistral") :
   rag_prompt_template = PromptTemplate(template=mistral_rag_prompt,input_variables=["context","question"], )


# Define LLM Model and Parameters 
# torch_dtype defines to run the model is full precision [32 bit] , half precision [ 16 bit ] 
# device_map setting to auto uses the accelerate library to distribute model acrss GPUs. 
# max_new_tokens defines the amount of tokens [ size of the output ] 
# Currently supported Model tasks are text-generation , text2text-generation , summarization , translation. 
# Attention - Each model is specific tunned for a specific task. 
model_task="text-generation"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto", torch_dtype=torch.float16)
#generation_config,unused_kwargs = GenerationConfig.from_pretrained( llm_model_name, do_sample=True, return_unused_kwargs=True, max_time=90)
#llm_model.generation_config = generation_config
pipe = pipeline(model_task, tokenizer=tokenizer, model=llm_model, max_new_tokens=4000, )
hf_pipe = HuggingFacePipeline(pipeline=pipe)


#Define Langchain Executor Framework. 
#Prepares the Input variables [ context , question ]
#Generates the Input Prompt using the prompt template
#Finally invokes the llm model with the input 
rag_chain = (
        { "context": retriever , "question": RunnablePassthrough() }
        | rag_prompt_template
        | hf_pipe
        )

#Define the LangServe with FastAPI 
app = FastAPI(
    title="Discover LangChain Server",
    version="1.0",
    description="An api server using Langchain's Runnable interfaces",
)

#Add the API Path to call the Langchain Executor framework 
add_routes(
    app,
    rag_chain ,
    path="/ragQuery",)


# Run the Model as API Server 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)