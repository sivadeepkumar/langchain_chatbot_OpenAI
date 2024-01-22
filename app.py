# from flask import Flask, request, jsonify, g
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.vectorstores.faiss import FAISS
# from langchain_community.llms.openai import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from dotenv import load_dotenv
# from flask_cors import CORS 
# from flask_sqlalchemy import SQLAlchemy
# import os
# import logging

# load_dotenv()

# app = Flask(__name__)
# CORS(app)


# # Replace the following line with your MySQL database URI
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://sivadeepkumar:1234@localhost/your_database'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

# # Load extracted text
# with open('extracted_text.txt', 'r') as f:
#     texts = f.read()

# embeddings = OpenAIEmbeddings()  
# document_search = FAISS.from_texts([texts], embeddings)
# chain = load_qa_chain(OpenAI(), chain_type="stuff")

# # # SQLite database setup
# db_path = 'your_database.db'  # Change to your desired database name


# # Define your models using SQLAlchemy
# class TagName(db.Model):
#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     chat_name = db.Column(db.String(50), nullable=False)

# class QAns(db.Model):
#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     chat_name_id = db.Column(db.Integer, db.ForeignKey('chat_name.id'), nullable=False)
#     query = db.Column(db.String(255), nullable=False)
#     response = db.Column(db.String(5000), nullable=True)

# # Rest of your code remains unchanged

# # Configure logging
# logging.basicConfig(filename='app.log', level=logging.INFO)
# logging.basicConfig(filename='app.log', level=logging.ERROR)


# # Function to get the database connection
# def get_db():
#     db = getattr(g, '_database', None)
#     if db is None:
#         db = g._database = db = SQLAlchemy(app)
#     return db


# @app.route('/new_chat', methods=['POST'])
# def new_chat():
#     try:
#         data = request.get_json()
#         chat_name = data.get('chat_name')

#         # If chat_name is not provided or an empty string, use the default value
#         chat_name = chat_name or 'New chat'

#         # Check if chat_name already exists in the database
#         db = get_db()
#         cursor = db.cursor()

#         cursor.execute('''
#             SELECT COUNT(*) FROM chat_name WHERE chat_name = ?
#         ''', (chat_name,))
#         count = cursor.fetchone()[0]

#         if count > 0:
#             return jsonify({'error': f'Tag name "{chat_name}" already exists. Please choose another tag name.'})

#         # Insert a new chat_name into the database (using the 'chat_name' table)
#         cursor.execute('''
#             INSERT INTO chat_name (chat_name)
#             VALUES (?)
#         ''', (chat_name,))
#         db.commit()

#         return jsonify({'message': 'New chat created successfully'})
#     except Exception as e:
#         return jsonify({'error': str(e)})
    
# @app.route('/query', methods=['POST'])
# def query():
#     data = request.get_json()
#     query = data['query']
#     chat_name = data["chat_name"]
    
#     result = "AI stands for artificial intelligence. It is a branch of computer science that focuses on developing machines and systems that can perform tasks that typically require human intelligence, such as problem solving, decision making, and pattern recognition. AI involves the use of algorithms and data to enable machines to learn from and adapt to new situations, making them more efficient and effective at completing tasks."
#     # Check if chat_name exists in the 'chat_name' table
#     db = get_db()
#     cursor = db.cursor()
#     cursor.execute('''
#         SELECT id FROM chat_name
#         WHERE chat_name = ?
#     ''', (query,))
#     chat_name_data = cursor.fetchone()

#     if not chat_name_data:
#         # If chat_name doesn't exist, create a new record in the 'chat_name' table
#         cursor.execute('''
#             INSERT INTO chat_name (chat_name)
#             VALUES (?)
#         ''', (query,))
#         db.commit()

#         # Get the latest chat_name id
#         cursor.execute('SELECT MAX(id) FROM chat_name')
#         latest_tag_id = cursor.fetchone()[0]
#     else:
#         # If chat_name exists, use its id
#         latest_tag_id = chat_name_data[0]

#     # Insert data into the 'q_ans' table with the latest chat_name
#     cursor.execute('''
#         INSERT INTO q_ans (chat_name, query, response)
#         VALUES (?, ?, ?)
#     ''', (latest_tag_id, query, result))
#     db.commit()

#     # # Retrieve all data associated with the specified chat_name from the 'q_ans' table
#     # cursor.execute('''
#     #     SELECT q.query, q.response
#     #     FROM q_ans q
#     #     JOIN chat_name t ON q.chat_name_id = t.id
#     #     WHERE t.chat_name = ?
#     # ''', (chat_name,))
#     # data = cursor.fetchall()

#     # # Convert the data to a list of dictionaries for JSON response
#     # result_data = []
#     # for row in data:
#     #     result_data.append({
#     #         'question': row[0],
#     #         'answer': row[1]
#     #     })

#     # return jsonify(result_data)

#     return jsonify({"question":query,"answer": result })


# @app.route('/delete', methods=['POST'])
# def delete_data():
#     data = request.get_json()
#     chat_name_to_delete = data.get('chat_name', '')

#     db = get_db()
#     cursor = db.cursor()

#     try:
#         # Delete data from the 'q_ans' table based on chat_name
#         cursor.execute('''
#             DELETE FROM q_ans
#             WHERE chat_name = ?
#         ''', (chat_name_to_delete,))

#         # Delete data from the 'chat_name' table based on chat_name
#         cursor.execute('''
#             DELETE FROM chat_name
#             WHERE chat_name = ?
#         ''', (chat_name_to_delete,))

#         db.commit()

#         return jsonify({'message': 'Data deleted successfully'})
#     except Exception as e:
#         # Log the exception using the logging module
#         logging.error(f"Error deleting data: {e}")
#         return jsonify({'error': 'An error occurred while deleting data'}), 500


# @app.route('/retrieve', methods=['GET'])
# def retrieve_data():
#     try:
#         chat_name_to_retrieve = request.args.get('chat_name', '')
#         db = get_db()
#         cursor = db.cursor()

#         if chat_name_to_retrieve:
#             # Retrieve data from the 'q_ans' table based on chat_name using JOIN
#             cursor.execute('''
#                 SELECT q.chat_name, q.query, q.response
#                 FROM q_ans q
#                 JOIN chat_name t ON q.chat_name = t.chat_name
#                 WHERE t.chat_name = ?
#             ''', (chat_name_to_retrieve,))
#             data = cursor.fetchall()

#             # Convert the data to a list of dictionaries for JSON response
#             result = []
#             for row in data:
#                 result.append({
#                     'chat_name': row[0],
#                     'query': row[1],
#                     'response': row[2]
#                 })

#             return jsonify(result)
#         else:
#             # If chat_name is not provided, retrieve all data from the 'q_ans' table
#             cursor.execute('''
#                 SELECT q.chat_name, q.query, q.response
#                 FROM q_ans q
#                 JOIN chat_name t ON q.chat_name = t.chat_name
#             ''')
#             data = cursor.fetchall()

#             # Convert the data to a list of dictionaries for JSON response
#             result = []
#             for row in data:
#                 result.append({
#                     'chat_name': row[0],
#                     'query': row[1],
#                     'response': row[2]
#                 })

#             return jsonify(result)
#     except Exception as e:
#         # Log the error
#         logging.error(f'Error retrieving data: {str(e)}')
#         return jsonify({'error': str(e)})


# @app.route('/update_chat_name', methods=['POST'])
# def update_chat_name():
#     try:
#         data = request.get_json()
#         old_chat_name = data.get('old_chat_name', '')
#         new_chat_name = data.get('new_chat_name', '')

#         db = get_db()
#         cursor = db.cursor()

#         # Check if the old_chat_name exists in the 'chat_name' table
#         cursor.execute('''
#             SELECT COUNT(*)
#             FROM chat_name
#             WHERE chat_name = ?
#         ''', (old_chat_name,))

#         count = cursor.fetchone()[0]

#         if count == 0:
#             raise ValueError(f'Tag name "{old_chat_name}" not found.')

#         # Update data in the 'chat_name' table based on old_chat_name
#         cursor.execute('''
#             UPDATE chat_name
#             SET chat_name = ?
#             WHERE chat_name = ?
#         ''', (new_chat_name, old_chat_name))

#         # Update data in the 'q_ans' table based on old_chat_name
#         cursor.execute('''
#             UPDATE q_ans
#             SET chat_name = ?
#             WHERE chat_name = ?
#         ''', (new_chat_name, old_chat_name))

#         db.commit()

#         # Log the update
#         logging.info(f'Tag name "{old_chat_name}" updated to "{new_chat_name}".')

#         return jsonify({'message': 'Data updated successfully'})
#     except Exception as e:
#         # Log the error
#         logging.error(f'Error updating tag name: {str(e)}')
#         return jsonify({'error': str(e)})
    
# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True)












from flask import Flask, request, jsonify, g
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS 
from langchain_community.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from flask_cors import CORS 
import os
import sqlite3
import logging

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
CORS(app)

# Load extracted text
with open('extracted_text.txt', 'r') as f:
    texts = f.read()

embeddings = OpenAIEmbeddings()  
document_search = FAISS.from_texts([texts], embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

# SQLite database setup
db_path = 'your_database.db'  # Change to your desired database name


# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO)
logging.basicConfig(filename='app.log', level=logging.ERROR)


def get_db():
    conn = sqlite3.connect('your_database.db')  # Replace 'your_database.db' with your actual database file
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/new_chat', methods=['POST'])
def new_chat():
    try:
        data = request.get_json()
        chat_id = data.get('id')

        # Check if the provided id exists in the 'chat_name' table
        db = get_db()
        cursor = db.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM chat_name WHERE id = ?
        ''', (chat_id,))
        count = cursor.fetchone()[0]

        if count > 0:
            return jsonify({'error': f'Chat with id "{chat_id}" already exists. Please choose another id.'})

        # Insert a new chat_name into the database (using the 'chat_name' table)
        cursor.execute('''
            INSERT INTO chat_name (id, chat_name)
            VALUES (?, ?)
        ''', (chat_id, f'New chat {chat_id}'))
        db.commit()

        return jsonify({'message': f'New chat with id {chat_id} created successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

    
# docs = document_search.similarity_search(query)
    # result = chain.run(input_documents=docs, question=query)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query = data['query']
    id = data["id"]
    
    result = "AI stands for artificial intelligence. It is a branch of computer science that focuses on developing machines and systems that can perform tasks that typically require human intelligence, such as problem-solving, decision-making, and pattern recognition. AI involves the use of algorithms and data to enable machines to learn from and adapt to new situations, making them more efficient and effective at completing tasks."

    # Check if chat_name exists in the 'chat_name' table
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT chat_name FROM chat_name
        WHERE id = ?
    ''', (id,))
    chat_id = cursor.fetchone()

    if not chat_id:
        # If chat_name doesn't exist, create a new record in the 'chat_name' table
        cursor.execute('''
            INSERT INTO chat_name (id, chat_name)
            VALUES (?, ?)
        ''', (id, query))
        db.commit()

        # Get the latest chat_name id
        cursor.execute('SELECT MAX(id) FROM chat_name')
        latest_tag_id = cursor.fetchone()[0]
    else:
        # If chat_name exists, use its id
        latest_tag_id = chat_id[0]

    # Insert data into the 'q_ans' table with the latest chat_name
    cursor.execute('''
        INSERT INTO q_ans (chat_name, query, response)
        VALUES (?, ?, ?)
    ''', (latest_tag_id, query, result))
    db.commit()

    return jsonify({"question": query, "answer": result})


@app.route('/delete', methods=['POST'])
def delete_data():
    data = request.get_json()
    chat_id_to_delete = data.get('id', '')

    db = get_db()
    cursor = db.cursor()

    try:
        # Retrieve chat_name based on id
        cursor.execute('''
            SELECT chat_name FROM chat_name
            WHERE id = ?
        ''', (chat_id_to_delete,))
        chat_name = cursor.fetchone()

        if chat_name:
            chat_name = chat_name[0]

            # Delete data from the 'q_ans' table based on chat_name
            cursor.execute('''
                DELETE FROM q_ans
                WHERE chat_name = ?
            ''', (chat_name,))

            # Delete data from the 'chat_name' table based on id
            cursor.execute('''
                DELETE FROM chat_name
                WHERE id = ?
            ''', (chat_id_to_delete,))

            db.commit()

            return jsonify({'message': 'Data deleted successfully'})
        else:
            return jsonify({'error': f'Chat with id {chat_id_to_delete} not found'})

    except Exception as e:
        # Log the exception using the logging module
        logging.error(f"Error deleting data: {e}")
        return jsonify({'error': 'An error occurred while deleting data'}), 500



@app.route('/individual_chat', methods=['POST'])
def individual_chat():
    try:
        data = request.get_json()
        chat_id_to_retrieve = data.get('id', '')
        db = get_db()
        cursor = db.cursor()

        # Retrieve data from the 'q_ans' table based on chat_id using JOIN
        cursor.execute('''
            SELECT t.id as id, q.chat_name, q.query, q.response
            FROM q_ans q
            JOIN chat_name t ON q.chat_name = t.chat_name
            WHERE t.id = ?
        ''', (chat_id_to_retrieve,))
        data = cursor.fetchall()

        # Check if there is any data for the given id
        if not data:
            return jsonify({'error': f'Chat ID "{chat_id_to_retrieve}" not found in the database.'})

        # Convert the data to a list of dictionaries for JSON response
        result = {'id': data[0][0], 'chat_name': data[0][1], 'messages': []}
        for row in data:
            result['messages'].append({
                'query': row[2],
                'response': row[3]
            })

        return jsonify(result)

    except Exception as e:
        # Log the error
        logging.error(f'Error retrieving data: {str(e)}')
        return jsonify({'error': str(e)})





@app.route('/all_chat', methods=['GET'])
def all_chat():
    try:
        db = get_db()
        cursor = db.cursor()

        # Retrieve all data from the 'q_ans' table
        cursor.execute('''
            SELECT t.id as id, q.chat_name, q.query, q.response
            FROM q_ans q
            JOIN chat_name t ON q.chat_name = t.chat_name
        ''')
        data = cursor.fetchall()

        # Organize data into a dictionary to consolidate by chat_name
        consolidated_data = {}
        for row in data:
            chat_name = row[1]
            if chat_name not in consolidated_data:
                consolidated_data[chat_name] = {
                    'id': row[0],
                    'chat_name': chat_name,
                    'messages': []
                }
            consolidated_data[chat_name]['messages'].append({
                'query': row[2],
                'response': row[3]
            })

        # Convert the organized data to a list for JSON response
        result = list(consolidated_data.values())

        return jsonify(result)

    except Exception as e:
        # Log the error
        logging.error(f'Error retrieving all chat data: {str(e)}')
        return jsonify({'error': str(e)})



@app.route('/update_chat_name', methods=['POST'])
def update_chat_name():
    try:
        data = request.get_json()
        old_chat_name = data.get('old_chat_name', '')
        new_chat_name = data.get('new_chat_name', '')

        db = get_db()
        cursor = db.cursor()

        # Check if the old_chat_name exists in the 'chat_name' table
        cursor.execute('''
            SELECT COUNT(*)
            FROM chat_name
            WHERE chat_name = ?
        ''', (old_chat_name,))

        count = cursor.fetchone()[0]

        if count == 0:
            raise ValueError(f'Tag name "{old_chat_name}" not found.')

        # Update data in the 'chat_name' table based on old_chat_name
        cursor.execute('''
            UPDATE chat_name
            SET chat_name = ?
            WHERE chat_name = ?
        ''', (new_chat_name, old_chat_name))

        # Update data in the 'q_ans' table based on old_chat_name
        cursor.execute('''
            UPDATE q_ans
            SET chat_name = ?
            WHERE chat_name = ?
        ''', (new_chat_name, old_chat_name))

        db.commit()

        # Log the update
        logging.info(f'Tag name "{old_chat_name}" updated to "{new_chat_name}".')

        return jsonify({'message': 'Data updated successfully'})
    except Exception as e:
        # Log the error
        logging.error(f'Error updating tag name: {str(e)}')
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)




