import requests
import sys
import json
import unidecode
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List

file_path = sys.argv[1]

loader = PyPDFLoader(file_path=file_path)
data = loader.load()

cleaned_data: List[Document] = []
for doc in data:
  cleaned_data.append(Document(page_content=doc.page_content.replace('\n',' '), metadata=doc.metadata))

for doc in cleaned_data :
  k = doc.page_content
  k = unidecode.unidecode(k)

  response = requests.post(
    "http://localhost:8000/dfsai/pdf-summarizer/invoke",
    json = {"input" : k }
  )

  j = response.json()
  print(j['output']['output'])

#print(type(j))

#for itm in j :
#    print(j[itm])

