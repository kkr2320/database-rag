from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

model = Ollama(model="llama3")

file_path = "d77cf8fd-03ce-4fa5-a09c-bcb4e8d2e6c7.pdf"

loader = PyPDFLoader(file_path=file_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

data = loader.load_and_split(text_splitter=text_splitter)

print(data.page_content)

template = """Format this : {question}"""

prompt = ChatPromptTemplate.from_template(template);

chain = (
        prompt
        | model
        | StrOutputParser()
        )
print(chain.invoke(data.page_content))
