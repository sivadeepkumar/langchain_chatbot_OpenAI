from flask import Flask, request, jsonify, g
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS 
from langchain_community.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
# from langchain.text_splitter import CharacterTextSplitter

from dotenv import load_dotenv
from flask_cors import CORS 
import os
# import sqlite3
import logging

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return 'OK', 200


@app.route('/cryoport', methods=['POST'])
def cryoport():
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


@app.route('/realEstateQuery', methods=['POST'])
def realEstateQuery():
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

@app.route('/query', methods=['POST'])
def assetpanda():
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

@app.route('/webkorps_query', methods=['POST'])
def webkorps_query():
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


if __name__ == '__main__':
    app.run(debug=True)

























