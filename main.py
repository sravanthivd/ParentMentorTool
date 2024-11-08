import os
import pickle
import time
#from openai import OpenAI
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI

#llm = ChatGoogleGenerativeAI(model="gemini-pro")


from dotenv import load_dotenv
load_dotenv() # take environment variables from .env

from google import generativeai as genai

# Configure the Google Generative AI API key
API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=API_KEY)
if API_KEY is None:
    st.error("GEMINI_API_KEY not found in environment variables.")
else:
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

st.title("ParentMentor Tool")
st.sidebar.title("Parenting Article Urls")
urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"Url {i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Process Urls")
#create a pickle file to store vector database
file_path="faiss_store_openai.pkl"

main_placefolder = st.empty()
if process_url_clicked:
    #loading data
    loader=UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading Started....")
    data=loader.load() # 'data' will have all the processed data from urls.


    #split the data
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", ","],chunk_size=1000)
    main_placefolder.text("Data splitter Started....")
    chunks = text_splitter.split_documents(data) #in chunks we get all individual chunks

    #create embeddings and save in Faiss index
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_openai=FAISS.from_documents(chunks,embeddings)
    main_placefolder.text("Embedding Vector Started Building...")
    time.sleep(2)

    #Save the FAISS index to a pickle file
    with open(file_path,"wb") as f:
        pickle.dump(vectorstore_openai,f)

query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            #result will have dictionary like {"answer": " ","source":""}
            st.header("Answer")
            st.subheader(result["answer"])
            #display sources if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                source_list = sources.split("\n")
                for source in source_list:
                    st.write(source)



