import os
import sys
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_community.embeddings  import HuggingFaceBgeEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto")
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=400 ,
)
model = HuggingFacePipeline(pipeline=pipe)

connection_string = "postgresql+psycopg://pgadm:pgadm1234#@p1-east.cluster-cafary0vpprp.us-east-1.rds.amazonaws.com:5432/db1"


vectorstore = PGVector(
     embeddings=embeddings,
     collection_name="my_collection",
     connection=connection_string,
     distance_strategy = DistanceStrategy.COSINE,
     use_jsonb=True,
 )

#print(vectorstore.similarity_search(sys.argv[1], k=3))

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
