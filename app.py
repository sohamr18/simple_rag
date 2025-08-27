import json
import os
import sys
import yaml
import boto3
from dotenv import load_dotenv
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

from utils.utils import create_prompt_template, get_final_prompt, rag_llama
from utils.data_ingestion import data_ingestion, get_vector_store


# Load env vars
load_dotenv()
aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")
## Bedrock Clients
bedrock = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name="us-west-2",
)


def main():

    # Read config
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Define models
    model_id = config["model_id"]
    bedrock_embeddings = BedrockEmbeddings(
        model_id=config["embedding_model"], client=bedrock
    )

    # Create UI
    st.set_page_config("Chat PDF")

    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    print(user_question)
    with st.sidebar:
        st.title("Update Or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                vectorstore_faiss = get_vector_store(docs, bedrock_embeddings)
                st.success("Done")

    if st.button("Ask me"):
        with st.spinner("Processing..."):
            # Load PDF
            docs = data_ingestion()
            vectorstore_faiss = get_vector_store(docs, bedrock_embeddings)
            # Create Prompt
            prompt_template = create_prompt_template()
            final_prompt = get_final_prompt(
                prompt_template, vectorstore_faiss, user_question
            )
            # Retrieve
            st.write(rag_llama(bedrock, model_id, final_prompt))
            st.success("Done")


if __name__ == "__main__":
    main()
