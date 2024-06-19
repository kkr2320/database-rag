import requests
import sys
import json


#Using Requests 
#payload = { "question":  sys.argv[1] , "vectordb_args" : "to_char(doc_date,'YYYY') = '2024'" }
#payload = { "question":  sys.argv[1] }
if len(sys.argv) == 3 : 
   payload = { "question":  sys.argv[1] , "vectordb_args" : sys.argv[2] }
else : 
   payload = { "question":  sys.argv[1] }
   

print(payload)

response = requests.post(
    "http://localhost:8000/dfsai/finance-assistant/invoke",
    json={"input": payload}
)

print(response)

j = response.json()

print (j['output']['output'])



# using LangServe Remote Runnabble
#from langserve import RemoteRunnable

#chain = RemoteRunnable("http://localhost:8000/dfsai/finance-assistant/")

#res = chain.invoke({'question' : sys.argv[1], 'vectordb_args' : 'doc_report_date="03/24/2023"' })

#res = chain.invoke({'question' : sys.argv[1] }) 
#print(chain.invoke({'question' : sys.argv[1] }) )

#For invoke
#print(res.get('output'))

#print(res)
