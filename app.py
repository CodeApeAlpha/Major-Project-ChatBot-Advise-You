import os
from flask import Flask, request, render_template, jsonify
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Chroma

app = Flask(__name__)

# Load the API key from the environment variable set in Render
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True

query = None

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    # loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
    loader = DirectoryLoader("KnowledgeBase/")
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []

@app.route('/')
def chat_page():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def handle_message():
    global query
    message = request.json['message']

    if message in ['quit', 'q', 'exit']:
        return jsonify({'response': 'Goodbye!'})
    
    result = chain({"question": message, "chat_history": chat_history})
    chat_history.append((message, result['answer']))
    
    return jsonify({'response': result['answer']})

if __name__ == '__main__':
    app.run(debug=False, host ='0.0.0.0')
