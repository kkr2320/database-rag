from openai import OpenAI
import os , json
import sys
import numpy as np
import psycopg
import requests
from typing import List
from pgvector.psycopg import register_vector

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

question = sys.argv[1] 
vectordb_args = sys.argv[2]
vectordb_user = sys.argv[3]
vectordb_pass = sys.argv[4]

embed_url = "http://localhost:8002/dfsai/embedtext/invoke"

## VectorDB Search 
# Define Custom DB Vector Retreiver
def get_relevant_documents(question: str , vectordb_args: str ) -> List[str]:

     response = requests.post( embed_url, json={"input": question})
     embedding = response.json()
     embed_vector = np.array(embedding['output'])

     connection_string = "postgresql://{us1}:{pw1}@p1-east.cluster-cafary0vpprp.us-east-1.rds.amazonaws.com:5432/db1".format(us1=vectordb_user , pw1=vectordb_pass)

     with psycopg.connect(connection_string) as conn:
       register_vector(conn)
       sql_stmt = "select 'Reported Date : ' || s.doc_date || ' ' || doc_page_content from ( select doc_date , doc_page_content , row_number() over ( partition by doc_date order by embeddings <=> %s ) as cosine_similarity from dfs_financial_documents <VECTORDB_WHERE_CLAUSE> ) s where s.cosine_similarity < 3 order by s.doc_date;"
       if vectordb_args is not None :
          sql_stmt = sql_stmt.replace("<VECTORDB_WHERE_CLAUSE>", " WHERE " + vectordb_args + " ")
       else :
          sql_stmt = sql_stmt.replace("<VECTORDB_WHERE_CLAUSE>", " ")

       #sql_stmt = sql_stmt + "ORDER BY embeddings <=> %s LIMIT 5 "
       print("Vector DB Query " + sql_stmt)
       rslt = conn.execute(sql_stmt , (embed_vector,) )
       Doc = rslt.fetchall()

     return Doc

rag_prompt = """
     <|begin_of_text|><|start_header_id|>system<|end_header_id|>
     You are a Financial Application to answer financial related question from the context provided. Its important you only answer the question only based on the context supplied part of this request. Dont try to make up answer on your own. If the answer can not be generated , then respond back as "Non-Determittent". <|eot_id|>

      <|start_header_id|>user<|end_header_id|>
      Context: {context}

      Question: {question}
      <|eot_id|>

      <|start_header_id>assistant<|end_header_id|>
      """
rag_prompt = rag_prompt.format(question=question, context=get_relevant_documents(question,vectordb_args))

print(rag_prompt)


completion = client.completions.create(model=model, prompt=rag_prompt, max_tokens=5000 , stop = ["ENDA"])
print(completion)
comp_json = completion.choices[0].text
print("Completion result:", comp_json)
