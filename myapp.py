from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# List of PDF filenames
pdf_filenames = ["one.pdf","two.pdf","three.pdf","four.pdf","five.pdf",]   # "one.pdf","two.pdf","three.pdf","four.pdf","five.pdf","six.pdf","seven.pdf","eight.pdf","nine.pdf","ten.pdf","a.pdf","b.pdf","c.pdf","d.pdf","e.pdf","f.pdf","g.pdf","h.pdf","i.pdf","j.pdf","k.pdf","l.pdf"

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

        docs = document_search.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)

        # Do something with the result...
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)





