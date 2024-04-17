from flask import Flask, request, jsonify, g
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS 
from langchain_community.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain

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


@app.route('/forms', methods=['POST'])
def forms():
    """
    Generates a summary based on the input query and source.
    This function handles POST requests containing a JSON payload with 'query' and 'source' fields. It generates a summary by searching for relevant documents related to the provided source and answering the query using an AI-powered QA model. 
    The result is returned in JSON format.
    Returns:
        JSON: Summary generated based on the input query and source.
    """
    # Input query
    data = request.get_json()
    input_text = data['query'].lower()


    # <<<
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

    # >>>
    
    query_sub = f"provide me the relevant columns only for {form_name} would be"
    # Fine-tuning rule to ensure all columns are included
    prompt_engineering = """
    NOTE: Neverever try to return all the fields or columns always go with minimal fields related to it.Please try to follow this note.
    Example : I have n number of fields.assume in that 10 for medical. If i ask i need to create medical list then you need to provide me that 10 fields only.That easy it is.
    """
    
    # Append the fine-tuning rule to the query
    query = query_sub +"\n\n" + prompt_engineering

    source = data['source']


    embeddings = OpenAIEmbeddings()  
    document_search = FAISS.from_texts([source], embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)

    return jsonify(result)




if __name__ == '__main__':
    app.run(debug=True)
