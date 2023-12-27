from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdfs(pdf_filenames):
    raw_text = ""
    try:
        for pdf_filename in pdf_filenames:
            pdfreader = PdfReader(pdf_filename)
            for i, page in enumerate(pdfreader.pages):
                content = page.extract_text()
                if content:
                    raw_text += content
        return raw_text
    except Exception as e:
        logger.error(f"Error extracting text from PDFs: {str(e)}")
        return None

def split_text_into_chunks(raw_text):
    try:
        text_splitter = CharacterTextSplitter(
            separator="/n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        return text_splitter.split_text(raw_text)
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {str(e)}")
        return None

def setup_document_search(texts):
    try:
        embeddings = OpenAIEmbeddings()
        document_search = FAISS.from_texts(texts, embeddings)
        return document_search
    except Exception as e:
        logger.error(f"Error setting up document search: {str(e)}")
        return None

def setup_qa_chain():
    try:
        return load_qa_chain(OpenAI(), chain_type="stuff")
    except Exception as e:
        logger.error(f"Error setting up QA chain: {str(e)}")
        return None

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({"error": "Query not provided"}), 400

        raw_text = extract_text_from_pdfs(pdf_filenames)
        if raw_text is None:
            return jsonify({"error": "Failed to extract text from PDFs"}), 500

        texts = split_text_into_chunks(raw_text)
        if texts is None:
            return jsonify({"error": "Failed to split text into chunks"}), 500

        document_search = setup_document_search(texts)
        if document_search is None:
            return jsonify({"error": "Failed to set up document search"}), 500

        chain = setup_qa_chain()
        if chain is None:
            return jsonify({"error": "Failed to set up QA chain"}), 500

        docs = document_search.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)

        # Do something with the result...
        return jsonify(result)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
