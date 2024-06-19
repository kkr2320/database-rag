from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

text = "This is a test document. This is test the vector size based on the text size"

query_result = embeddings.embed_query(text)

print(query_result[:2048])

print(len(query_result))
