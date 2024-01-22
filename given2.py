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








# from flask import Flask, request, jsonify, g
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.vectorstores.faiss import FAISS 
# from langchain_community.llms.openai import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from dotenv import load_dotenv
# from flask_cors import CORS 

# import os
# import sqlite3

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# app = Flask(__name__)
# CORS(app)

# # Load extracted text
# with open('extracted_text.txt', 'r') as f:
#     texts = f.read()

# embeddings = OpenAIEmbeddings()  
# document_search = FAISS.from_texts([texts], embeddings)

# chain = load_qa_chain(OpenAI(), chain_type="stuff")

# # SQLite database setup
# db_path = 'your_database.db'  # Change to your desired database name

# def get_db():
#     db = getattr(g, '_database', None)
#     if db is None:
#         db = g._database = sqlite3.connect(db_path)
#     return db

# @app.route('/query', methods=['POST'])
# def query():
#     data = request.get_json()
#     query = data['query']
    
#     docs = document_search.similarity_search(query)
#     result = chain.run(input_documents=docs, question=query)

#     # Insert data into the database
#     tag_name = query  # Assuming 'tag_name' is provided in the JSON data
#     response = result  # Assuming the answer is available in the 'answer' field of the result

#     db = get_db()
#     cursor = db.cursor()

#     cursor.execute('''
#         INSERT INTO your_table_name (tag_name, query, response)
#         VALUES (?, ?, ?)
#     ''', (tag_name, query, response))
#     db.commit()

#     return jsonify(result)

# @app.route('/delete', methods=['POST'])
# def delete_data():
#     data = request.get_json()
#     tag_name_to_delete = data.get('tag_name', '')

#     db = get_db()
#     cursor = db.cursor()

#     # Delete data from the database based on tag_name
#     cursor.execute('''
#         DELETE FROM your_table_name
#         WHERE tag_name = ?
#     ''', (tag_name_to_delete,))
#     db.commit()

#     return jsonify({'message': 'Data deleted successfully'})

# @app.route('/retrieve', methods=['GET'])
# def retrieve_data():
#     tag_name_to_retrieve = request.args.get('tag_name', '')
#     db = get_db()
#     cursor = db.cursor()

#     if tag_name_to_retrieve:
#         # Retrieve data from the database based on tag_name
#         cursor.execute('''
#             SELECT * FROM your_table_name
#             WHERE tag_name = ?
#         ''', (tag_name_to_retrieve,))
#         data = cursor.fetchall()

#         # Convert the data to a list of dictionaries for JSON response
#         result = []
#         for row in data:
#             result.append({
#                 'id': row[0],
#                 'tag_name': row[1],
#                 'query': row[2],
#                 'response': row[3]
#             })

#         return jsonify(result)
#     else:
#         # If tag_name is not provided, retrieve all data
#         cursor.execute('SELECT * FROM your_table_name')
#         data = cursor.fetchall()

#         # Convert the data to a list of dictionaries for JSON response
#         result = []
#         for row in data:
#             result.append({
#                 'id': row[0],
#                 'tag_name': row[1],
#                 'query': row[2],
#                 'response': row[3]
#             })

#         return jsonify(result)


# @app.route('/update_tag_name', methods=['POST'])
# def update_tag_name():
#     data = request.get_json()
#     old_tag_name = data.get('old_tag_name', '')  # Add a new field for the old tag name
#     new_tag_name = data.get('new_tag_name', '')  # Add a new field for the new tag name

#     db = get_db()
#     cursor = db.cursor()

#     # Update data in the database based on old_tag_name
#     cursor.execute('''
#         UPDATE your_table_name
#         SET tag_name = ?
#         WHERE tag_name = ?
#     ''', (new_tag_name, old_tag_name))
#     db.commit()

#     return jsonify({'message': 'Data updated successfully'})





# if __name__ == '__main__':
#     app.run(debug=True)