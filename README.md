# ParentMentorTool
ParentMetor Tool is a web-based application designed to assist new parents in navigating early parenting challenges. By allowing users to input trusted parenting Urls, the tool provide contextually relevant answers  to baby-related queries, along with the list of sources from where the information is originated.

![image](https://github.com/user-attachments/assets/686c3942-4d51-424b-8aa0-664840963459)
## Features
- Load URLs or upload text files containing URLs to fetch article content.
- Process article content through LangChain's UnstructuredURL Loader
- Construct an embedding vector using Huggingface's embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information
- Interact with the google LLM's by inputting queries and receiving answers along with source URLs.

## Project Structure
- main.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.
- faiss_store_openai.pkl: A pickle file to store the FAISS index.
- env: Configuration file for storing your google API key.
## Usage
1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.
- On the sidebar, you can input URLs directly.

- Initiate the data loading and processing by clicking "Process URLs."

- Observe the system as it performs text splitting, generates embedding vectors, and efficiently indexes them using FAISS.

- The embeddings will be stored and indexed using FAISS, enhancing retrieval speed.

- The FAISS index will be saved in a local file path in pickle format for future use.

- One can now ask a question and get the answer based on those news articles
