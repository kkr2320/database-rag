import os
import sys
import numpy as np 
import psycopg
from pgvector.psycopg import register_vector
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_community.embeddings  import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi import FastAPI
from langserve import add_routes


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

#model_name = "nomic-ai/nomic-embed-text-v1"
model_name = "Snowflake/snowflake-arctic-embed-l"
model_kwargs = {
    'device': "cuda",
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

model_id="meta-llama/Meta-Llama-3-8B-Instruct"
#model_id="google/flan-t5-xl"
#model_id="meta-llama/Llama-2-7b-chat-hf"
#model_id="mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto")
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1000 ,
)
model = HuggingFacePipeline(pipeline=pipe)

class vectordb_fetch:
   def __call__(self,question):
     question_array = embeddings.embed_query(question) 

     question_vectors = np.array(question_array)

#    print(question_vectors)

     connection_string = "postgresql://pgadm:pgadm1234#@p1-east.cluster-cafary0vpprp.us-east-1.rds.amazonaws.com:5432/db1"

     with psycopg.connect(connection_string) as conn: 
       register_vector(conn)
       rslt = conn.execute("SELECT document FROM langchain_pg_embedding ORDER BY embedding <=> %s LIMIT 5" , (question_vectors,) )
       Doc = rslt.fetchall()

     return Doc

retriever = vectordb_fetch()

rag_prompt = """Answer the question only based on the context below. Dont try to make your own answer. If the
question cannot be answered using the information provided answer with "I don't know".

Context: {context}

Question: {question}

Answer: """

rag_prompt_template = PromptTemplate(template=rag_prompt,input_variables=["context","question"]);

rag_chain = (
        { "context": retriever , "question": RunnablePassthrough() }
        | rag_prompt_template
        | model
        )
print(rag_chain.invoke(sys.argv[1]))
