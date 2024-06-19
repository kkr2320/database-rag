from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_community.embeddings  import HuggingFaceBgeEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration


model_name = "nomic-ai/nomic-embed-text-v1"
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

model_id="google/flan-t5-xl"
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)
pipe = pipeline("summarization", model=model, tokenizer=tokenizer, max_new_tokens=2048,max_length=2048)
hf = HuggingFacePipeline(pipeline=pipe)

connection_string = "postgresql+psycopg://pgadm:pgadm1234#@p1-east.cluster-cafary0vpprp.us-east-1.rds.amazonaws.com:5432/db1"

vectorstore = PGVector(
     embeddings=embeddings,
     collection_name="my_collection",
     connection=connection_string,
      distance_strategy = DistanceStrategy.COSINE,
     use_jsonb=True,
 )

ctx = vectorstore.similarity_search("QuantumScape 2023 Financial statements", k=3)

rag_template = """Summarize this with minimum 1000 characters: {question}"""

rag_prompt = PromptTemplate(template=rag_template,input_variables=["question"]);

rag_chain = (
        rag_prompt
        | hf
        | StrOutputParser()
        )
print(rag_chain.invoke(ctx))

