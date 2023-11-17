from flask import Flask, Response, render_template, request, jsonify
from llama_index import SimpleDirectoryReader,GPTListIndex,GPTVectorStoreIndex,LLMPredictor,PromptHelper,ServiceContext,StorageContext,load_index_from_storage
from langchain import OpenAI
import sys
import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


app = Flask(__name__)
  
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
        input = request.form["msg"]
        create_index("knowledge")
        response = answerMe(input)
        return response
    
    
def create_index(path):
  max_input = 4096
  tokens = 200
  chunk_size = 600 #for LLM, we need to define chunk size
  max_chunk_overlap = 0.5

  promptHelper = PromptHelper(max_input,tokens,max_chunk_overlap,chunk_size_limit=chunk_size)  #define prompt   

  llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001",max_tokens=tokens)) #define LLM
  docs = SimpleDirectoryReader(path).load_data() #load data

  #create vector index

  service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor,prompt_helper=promptHelper)
    
  vectorIndex = GPTVectorStoreIndex.from_documents(documents=docs,service_context=service_context)
  vectorIndex.storage_context.persist(persist_dir = 'Essai')
  
def answerMe(question):
    storage_context = StorageContext.from_defaults(persist_dir='Essai')
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return response.response


if __name__ == '__main__':
    app.run()
