import os
import sys
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_community.embeddings  import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi import FastAPI
from langserve import add_routes
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
#model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto")
pipe = pipeline(
    "text2text-generation", model=model_id, tokenizer=tokenizer, max_new_tokens=10000 , device_map="auto",
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

response_schemas = [
    ResponseSchema(name="Answer", description="answer to the user's question"),
    ResponseSchema( name="Question", description="Question") ,
]
out_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = out_parser.get_format_instructions()

rag_prompt = """Answer the question only based on the context below. Dont try to make your own answer. If the
question cannot be answered using the information provided answer with "I don't know".

Context: {context}

Question: {question}

Answer: """

rag_prompt_template = PromptTemplate(template=rag_prompt,input_variables=["context","question"],output_parser=out_parser )


rag_chain = (
        { "context" : retriever, "question": RunnablePassthrough() }
        | rag_prompt_template
        | model
        )

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    rag_chain ,
    path="/ragQuery",)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
