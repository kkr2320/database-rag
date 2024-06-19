import os
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

hf = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloomz-7b1",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)

template = """Question: {question} """

prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "Translate English to Tamil : Couples Living Together "

print(chain.invoke({"question": question}))
