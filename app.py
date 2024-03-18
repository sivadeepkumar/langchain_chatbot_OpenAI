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




























# from flask import Flask, request, jsonify
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.vectorstores.faiss import FAISS 
# from langchain_community.llms.openai import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from dotenv import load_dotenv
# from flask_cors import CORS 
# import os
# import logging

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# app = Flask(__name__)
# CORS(app)

# # Set up logging
# logging.basicConfig(filename='app.log', level=logging.INFO)

# def load_text(file_path):
#     try:
#         with open(file_path, 'r') as file:
#             return file.read()
#     except Exception as e:
#         logging.error(f"Failed to load text from {file_path}: {e}")
#         return None

# def create_chain(embeddings, texts):
#     try:
#         embeddings_instance = OpenAIEmbeddings()
#         document_search_instance = FAISS.from_texts([texts], embeddings_instance)
#         chain_instance = load_qa_chain(OpenAI(), chain_type="stuff")
#         return embeddings_instance, document_search_instance, chain_instance
#     except Exception as e:
#         logging.error(f"Failed to create chain: {e}")
#         return None, None, None

# # Load extracted text Asset panda
# texts = load_text('extracted_text.txt')
# if texts:
#     embeddings, document_search, chain = create_chain(OpenAIEmbeddings(), texts)

# # Load extracted text Webkorps LLC
# texts2 = load_text('webkorps_data.txt')
# if texts2:
#     embeddings2, document_search2, chain2 = create_chain(OpenAIEmbeddings(), texts2)

# @app.route('/query', methods=['POST'])
# def query():
#     try:
#         data = request.get_json()
#         query = data.get('query')

#         if not query:
#             raise ValueError("Invalid request: 'query' parameter is missing.")

#         docs = document_search.similarity_search(query)
#         result = chain.run(input_documents=docs, question=query)

#         return jsonify({"question": query, "answer": result})

#     except Exception as e:
#         logging.error(f"Error in /query endpoint: {e}")
#         return jsonify({"error": "Internal Server Error"}), 500

# @app.route('/webkorps_query', methods=['POST'])
# def webkorps_query():
#     try:
#         data = request.get_json()
#         query = data.get('query')

#         if not query:
#             raise ValueError("Invalid request: 'query' parameter is missing.")

#         docs = document_search2.similarity_search(query)
#         result = chain2.run(input_documents=docs, question=query)

#         return jsonify({"question": query, "answer": result})

#     except Exception as e:
#         logging.error(f"Error in /webkorps_query endpoint: {e}")
#         return jsonify({"error": "Internal Server Error"}), 500


# if __name__ == '__main__':
#     app.run(debug=True)



