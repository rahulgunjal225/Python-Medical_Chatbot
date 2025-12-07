from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from src.prompt import PROMPT_TEMPLATE
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.llms import HuggingFaceHub  
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "medical-chatbot"

# Initialize embeddings and vector store
print(" Loading HuggingFace Embeddings (MiniLM-L6-v2)...")
embeddings = download_hugging_face_embeddings()
vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

#  Use a free, public Hugging Face model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    task="text2text-generation",  
    model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
)



# Define prompt template
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# Create RAG pipeline
question_answer_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

# Updated route with debug info
@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    print(f" User asked: {user_input}")

    try:
        response = rag_chain.invoke({"input": user_input})
        print(" Full response:", response)
        answer = response.get("answer", "Sorry, I couldn’t find an answer for that.")
        return jsonify({"response": answer})
    except Exception as e:
        print("Error during chat:", e)
        return jsonify({"response": "Error: " + str(e)})

if __name__ == "__main__":
    app.run(debug=False)
