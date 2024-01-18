from flask import Flask, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS 
from langchain_community.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os
from flask_cors import CORS
load_dotenv()

# OPENAI_API_KEY =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"OPENAI_API_KEY")

app = Flask(__name__)
app.config['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
CORS(app)
# Load extracted text
with open('extracted_text.txt', 'r') as f:
    texts = f.read()

embeddings = OpenAIEmbeddings()  
document_search = FAISS.from_texts([texts], embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query = data['query']
    
    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)


