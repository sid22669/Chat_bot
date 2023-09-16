from flask import Flask, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS

app = Flask(__name__)

# Set your API key and CSV file path here
user_api_key = "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"
csv_file_path = "Course_data_2.csv"

# Load and process the CSV file
loader = CSVLoader(file_path=csv_file_path, encoding="utf-8")
data = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)

vectors = FAISS.from_documents(data, embeddings)

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=user_api_key),
    retriever=vectors.as_retriever()
)

# Function to interact with the chatbot
def conversational_chat(query):
    result = chain({"question": query, "chat_history": []})
    return result["answer"]

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_query = data["query"]
    bot_response = conversational_chat(user_query)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
