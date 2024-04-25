import json
import os
import sys
import boto3
from flask import Flask, request , jsonify

app = Flask(__name__)

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

bedrock = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client = bedrock)
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_pdf_from_string(file_path, data_string):
    # Create a canvas object
    c = canvas.Canvas(file_path, pagesize=letter)

    # Define font and size
    c.setFont("Helvetica", 12)

    # Split the data string into lines
    lines = data_string.split('\n')

    # Set initial y position for text
    y_position = 750

    # Write each line to the PDF
    for line in lines:
        c.drawString(100, y_position, line)
        y_position -= 20  # Move down 20 units for the next line

    # Save the PDF
    c.save()



def manual_ingestion(input_string):
    # Create a single document from the input string
    single_document = {"page_content": input_string}

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    # Split the single document into chunks
    chunks = text_splitter.split_documents([single_document])

    # Extract the text content from the chunks
    chunk_texts = [chunk['page_content'] for chunk in chunks]

    return chunk_texts





# manual_ingestion
def manual_vector_store(docs):
    # Assuming bedrock_embeddings is defined somewhere
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("manual_index")

from langchain.text_splitter import RecursiveCharacterTextSplitter

def manual_ingestion(text_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([text_data])
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
    llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock, model_kwargs={'max_tokens': 200})
    return llm

def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={'max_gen_len': 512})
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
    user_question = request.json['user_question']
    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    llm = get_mistral_llm()
    response = get_response_llm(llm, faiss_index, user_question)
    return response

# Route for Llama2 model response using POST request
@app.route('/llama2_response', methods=['POST'])
def llama2_response():
    user_question = request.json['user_question']
    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    llm = get_llama2_llm()
    response = get_response_llm(llm, faiss_index, user_question)
    return response


@app.route('/mistral/source', methods=['POST'])
def mistral_source():
    user_question = request.json['user_question']
    source_data = request.json['source'] 
    docs = manual_ingestion(source_data)
    manual_vector_store(docs)
    # Load the FAISS index
    faiss_index = FAISS.load_local("manual_index", bedrock_embeddings, allow_dangerous_deserialization=True)

    # Assuming these functions are defined elsewhere
    llm = get_llama2_llm()
    response = get_response_llm(llm, faiss_index, user_question)
    return jsonify({'response': response})


@app.route('/llama/source', methods=['POST'])
def llama_source():
    user_question = request.json['user_question']
    source_data = request.json['source'] 
    docs = manual_ingestion(source_data)
    manual_vector_store(docs)
    # Load the FAISS index
    faiss_index = FAISS.load_local("manual_index", bedrock_embeddings, allow_dangerous_deserialization=True)

    # Assuming these functions are defined elsewhere
    llm = get_llama2_llm()
    response = get_response_llm(llm, faiss_index, user_question)
    return jsonify({'response': response})


@app.route('/llama/form', methods=['POST'])
def llama_form():
    user_question = request.json['user_question']
    source_data = request.json['source']
    # <<<<<
    input_text = user_question.lower()
    
    query_words = input_text.split()
    index_of_form = [i for i, word in enumerate(query_words) if word == "form"]

    if index_of_form and index_of_form[0] > 0:
        form_left_word_index = index_of_form[0] - 1
        form_name = query_words[form_left_word_index] if form_left_word_index >= 0 else ""
    else:
        form_name = ""
        
    if form_name == "":
        result =  "Please provide in proper format"
        return jsonify(result)
    

    query_sub = f"provide me the relevant columns only for {form_name} would be"
    # Fine-tuning rule to ensure all columns are included
    prompt_engineering = """
    NOTE: Neverever try to return all the fields or columns always go with minimal fields related to it.Please try to follow this note.
    Example : I have n number of fields.assume in that 10 for medical. If i ask i need to create medical list then you need to provide me that 10 fields only.That easy it is.
    """
    
    # Append the fine-tuning rule to the query
    query = query_sub +"\n\n" + prompt_engineering

    # >>>>>
    # Process the source data by splitting into chunks
    
    docs = manual_ingestion(source_data)
    manual_vector_store(docs)
    # Load the FAISS index
    faiss_index = FAISS.load_local("manual_index", bedrock_embeddings, allow_dangerous_deserialization=True)

    # Assuming these functions are defined elsewhere
    llm = get_llama2_llm()
    response = get_response_llm(llm, faiss_index, query)
    return jsonify({'response': response})


@app.route('/mistral/form', methods=['POST'])
def mistral_form():
    user_question = request.json['user_question']
    source_data = request.json['source']
    # <<<<<
    input_text = user_question.lower()
    
    query_words = input_text.split()
    index_of_form = [i for i, word in enumerate(query_words) if word == "form"]

    if index_of_form and index_of_form[0] > 0:
        form_left_word_index = index_of_form[0] - 1
        form_name = query_words[form_left_word_index] if form_left_word_index >= 0 else ""
    else:
        form_name = ""
        
    if form_name == "":
        result =  "Please provide in proper format"
        return jsonify(result)
    

    query_sub = f"provide me the relevant columns only for {form_name} would be"
    # Fine-tuning rule to ensure all columns are included
    prompt_engineering = """
    NOTE: Neverever try to return all the fields or columns always go with minimal fields related to it.Please try to follow this note.
    Example : I have n number of fields.assume in that 10 for medical. If i ask i need to create medical list then you need to provide me that 10 fields only.That easy it is.
    """
    
    # Append the fine-tuning rule to the query
    query = query_sub +"\n\n" + prompt_engineering
    # import pdb ; pdb.set_trace()
    # >>>>>
    # Process the source data by splitting into chunks
    docs = manual_ingestion(source_data)
    manual_vector_store(docs)

    # Load the FAISS index
    faiss_index = FAISS.load_local("manual_index", bedrock_embeddings, allow_dangerous_deserialization=True)

    # Assuming these functions are defined elsewhere
    llm = get_llama2_llm()
    response = get_response_llm(llm, faiss_index, query)

    return jsonify({'response': response})






if __name__ == '__main__':
    app.run(debug=True)
