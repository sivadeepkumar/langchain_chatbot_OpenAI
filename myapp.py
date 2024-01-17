from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
from flask_cors import CORS 
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime 
# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)
#OPENAI ENV
app.config['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chats.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Chat(db.Model):
    __tablename__ = 'chat_table'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    chat_name = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    query = db.Column(db.String(255), nullable=False)
    result = db.Column(db.String(5000), nullable=True)

# List of PDF filenames
pdf_filenames = ["Groups_and_Records.pdf"]   

raw_text = ""

# Loop through PDFs and extract text
for pdf_filename in pdf_filenames:
    pdfreader = PdfReader(pdf_filename)
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

text_splitter = CharacterTextSplitter(
    separator="/n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

document_search = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        query = data['query']
        user_id = 1
        chat_name =  "chat_name" #data['chat_name']
        docs = document_search.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)

        # Store the chat in the database
        new_chat = Chat(user_id=user_id, chat_name=chat_name, query=query, result=result)
        db.session.add(new_chat)
        db.session.commit()
        
        # Do something with the result...
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


# When the user clicks on new chat or existing chat it have to filter for db and need to give data
@app.route('/get_data', methods=['POST'])
def get_data():
    try:
        import pdb 
        pdb.set_trace()
        data = request.get_json()
        seperate_chat = data['chat_name']

        # Query the database for the specified chat_name
        # chat_data = Chat.query.filter(Chat.chat_name == seperate_chat).all()
        chat_data2 = Chat.query.filter_by(chat_name=seperate_chat).all()
        # Format the data for response
        formatted_data = [{'timestamp': chat.timestamp, 'query': chat.query, 'result': chat.result} for chat in chat_data]
        return jsonify({'chat_data': formatted_data})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Create the database tables before running the app
    with app.app_context():
        db.create_all()
    app.run(debug=True)



