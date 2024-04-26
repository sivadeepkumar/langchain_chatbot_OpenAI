from flask import Flask, request, jsonify, g
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS 
from langchain_community.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3

# Initialize AWS services and models
region_name = 'us-east-1'  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.chains.question_answering import load_qa_chain

bedrock = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client = bedrock)

from dotenv import load_dotenv
from flask_cors import CORS 
import os
import logging
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return 'OK', 200


@app.route('/cryoport', methods=['POST'])
def cryoport():
    try:
        data = request.get_json()
        query = data['query']

        with open('cryoport_text.txt', 'r') as f:
            texts = f.read()
        embeddings = OpenAIEmbeddings()  
        document_search = FAISS.from_texts([texts], embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")


        docs = document_search.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)

        return jsonify(result)
    except Exception as e:
        logger.exception("An error occurred in /cryoport endpoint.")
        return jsonify({'error': 'Internal Server Error'}), 500


@app.route('/realEstateQuery', methods=['POST'])
def realEstateQuery():
    try:
        data = request.get_json()
        query = data['query']

        with open('estate.txt', 'r') as f:
            texts = f.read()
        embeddings = OpenAIEmbeddings()  
        document_search = FAISS.from_texts([texts], embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")


        docs = document_search.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)

        return jsonify(result)
    except Exception as e:
        logger.exception("An error occurred in /realEstateQuery endpoint.")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/query', methods=['POST'])
def assetpanda():
    try:
        data = request.get_json()
        query = data['query']

        with open('assetpanda.txt', 'r') as f:
            texts = f.read()
        embeddings = OpenAIEmbeddings()  
        document_search = FAISS.from_texts([texts], embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")


        docs = document_search.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)

        return jsonify(result)
    except Exception as e:
        logger.exception("An error occurred in /query (ASSETPANDA) endpoint.")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/webkorps_query', methods=['POST'])
def webkorps_query():
    try:
        data = request.get_json()
        query = data['query']

        with open('webkorps_data.txt', 'r') as f:
            texts = f.read()
        embeddings = OpenAIEmbeddings()  
        document_search = FAISS.from_texts([texts], embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        docs = document_search.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)

        return jsonify(result)
    except Exception as e:
        logger.exception("An error occurred in /webkorps_query endpoint.")
        return jsonify({'error': 'Internal Server Error'}), 500


@app.route('/summary', methods=['POST'])
def summary():
    data = request.get_json()
    query = data['query']
    source = data['source']


    embeddings = OpenAIEmbeddings()  
    document_search = FAISS.from_texts([source], embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)

    return jsonify(result)


def get_embedding(source,embeddings):
    query = "Which form we need to create just one word answer like <Create the ________ form> in this format,I need to get the response"
    document_search = FAISS.from_texts([source], embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    cleaned_string = result.replace('\n', '')

    return cleaned_string

@app.route('/openai/form', methods=['POST'])
def forms():
    """
    Generates a summary based on the input query and source.
    This function handles POST requests containing a JSON payload with 'query' and 'source' fields. It generates a summary by searching for relevant documents related to the provided source and answering the query using an AI-powered QA model. 
    The result is returned in JSON format.
    Returns:
        JSON: Summary generated based on the input query and source.
    """
    # Input query
    # import pdb ; pdb.set_trace()
    embeddings = OpenAIEmbeddings()
    data = request.get_json()
    query = data["query"]
    source = data['source']
    # print(query)
    # Method_2 
    
    query_words = get_embedding(query,embeddings)
    print(query_words)  
    query_sub = f"provide me the relevant columns only for {query_words} would be"
    # query_sub = f"provide me the relevant columns name in array only for {query} would be."
    # Fine-tuning rule to ensure all columns are included
    prompt_engineering = """
    NOTE: Neverever try to return all the fields or columns always go with minimal fields related to it.Please try to follow this note.
    Example : I have n number of fields.assume in that 10 for medical. If i ask i need to create medical list then you need to provide me that 10 fields only.That easy it is.
    """
    # pdb.set_trace()
    # Append the fine-tuning rule to the query
    query = query_sub +"\n\n" + prompt_engineering
    # query = f"Create the {query_words} form"
    document_search = FAISS.from_texts([source], embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)

    return jsonify(result)

# manual_ingestion
def manual_vector_store(docs):
    # Assuming bedrock_embeddings is defined somewhere
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("manual_index")


def manual_ingestion(source):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([source])
    return docs

# Data ingestion function
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector embedding and vector store function
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# LLM models functions
def get_mistral_llm():
    llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock, model_kwargs={'max_tokens': 1024})
    return llm

def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

# Prompt template for LLM responses
prompt_template = """
{context}
Question: {question}
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Function to get response from LLM model
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                     retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
                                     return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})
    answer = qa({"query": query})
    return answer['result']

# Route for updating vector base using GET request
@app.route('/update_vector_base', methods=['GET'])
def update_vector_base():
    docs = data_ingestion()
    get_vector_store(docs)
    return 'Vectors Updated Successfully'

# Route for Mistral model response using POST request
@app.route('/mistral_response', methods=['POST'])
def mistral_response():
    user_question = request.json['query']
    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    llm = get_mistral_llm()
    response = get_response_llm(llm, faiss_index, user_question)
    return response

# Route for Llama2 model response using POST request
@app.route('/llama2_response', methods=['POST'])
def llama2_response():
    user_question = request.json['query']
    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    llm = get_llama2_llm()
    response = get_response_llm(llm, faiss_index, user_question)
    return response



@app.route('/llama/source', methods=['POST'])
def llama_source():
    data = request.get_json()
    query = data['query']
    source = data['source']

    # Load embeddings model (replace 'Your_Embeddings_Model' with the actual name of your embeddings model)
    embeddings = bedrock_embeddings

    # Load LLM from Amazon Bedrock (replace 'get_llm' with the actual function for loading Amazon Bedrock LLM)
    llm = get_llama2_llm()

    # Create FAISS index from source data
    document_search = FAISS.from_texts([source], embeddings)

    # Assuming load_qa_chain is a function to load your QA chain
    chain = load_qa_chain(llm, chain_type="stuff")

    # Perform similarity search and run the QA chain
    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    return jsonify(result)

@app.route('/mistral/source', methods=['POST'])
def mistral_source():
    data = request.get_json()
    query = data['query']
    source = data['source']

    # Load embeddings model (replace 'Your_Embeddings_Model' with the actual name of your embeddings model)
    embeddings = bedrock_embeddings

    # Load LLM from Amazon Bedrock (replace 'get_llm' with the actual function for loading Amazon Bedrock LLM)
    llm = get_mistral_llm()

    # Create FAISS index from source data
    document_search = FAISS.from_texts([source], embeddings)

    # Assuming load_qa_chain is a function to load your QA chain
    chain = load_qa_chain(llm, chain_type="stuff")

    # Perform similarity search and run the QA chain
    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    return jsonify(result)

def get_first_embedding(source_data):
    user_question = "Which form we need to create just one word answer like <Create the ________ form> in this format,I need to get the response"
    docs = manual_ingestion(source_data)
    manual_vector_store(docs)
    faiss_index = FAISS.load_local("manual_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    llm = get_mistral_llm()
    response = get_response_llm(llm, faiss_index, user_question)
    return response

@app.route('/mistral/form', methods=['POST'])
def mistral_form():
    user_question = request.json['query']
    source_data = request.json['source']

    query = get_first_embedding(user_question)

    print(query)
    user_question = f"provide me the relevant columns name in array only for {query} would be."
    docs = manual_ingestion(source_data)
    manual_vector_store(docs)

    
    # FROM STORED DATA IT WILL RETRIEVE 
    faiss_index = FAISS.load_local("manual_index", bedrock_embeddings, allow_dangerous_deserialization=True)

    # Assuming these functions are defined elsewhere
    llm = get_mistral_llm()
    response = get_response_llm(llm, faiss_index, user_question)
    return jsonify({'response': response})


@app.route('/llama/form', methods=['POST'])
def llama_form():
    user_question = request.json['query']
    source_data = request.json['source']

    query = get_first_embedding(user_question)
    print(query)
    user_question = f"provide me the relevant columns name in array only for {query} would be."
    docs = manual_ingestion(source_data)
    manual_vector_store(docs)

    
    # FROM STORED DATA IT WILL RETRIEVE 
    faiss_index = FAISS.load_local("manual_index", bedrock_embeddings, allow_dangerous_deserialization=True)

    # Assuming these functions are defined elsewhere
    llm = get_llama2_llm()
    response = get_response_llm(llm, faiss_index, user_question)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)





