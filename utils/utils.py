import json
import os
import sys
import boto3
from dotenv import load_dotenv
import streamlit as st

## We will be suing Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def create_prompt_template():

    prompt_template = """

    Human: Use the following pieces of context to provide a 
    concise answer to the question at the end but use atleast summarize with 
    250 words without detailed explaantions or notes. If you don't know the answer, 
    just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context
    Question: {question}
    Assistant:"""

    return prompt_template


def get_final_prompt(prompt_template, vectorstore_faiss, query, k=3):

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    question = query
    retrieved_docs = vectorstore_faiss.similarity_search(question, k=k)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    final_prompt = PROMPT.format(context=context, question=question)
    return final_prompt


def rag_llama(client, model_id, final_prompt):
    # model_id = "us.meta.llama3-2-3b-instruct-v1:0"

    native_request = {
        "prompt": final_prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
    }
    request = json.dumps(native_request)

    response = client.invoke_model(modelId=model_id, body=request)
    model_response = json.loads(response["body"].read())

    response_text = model_response["generation"]

    return response_text

