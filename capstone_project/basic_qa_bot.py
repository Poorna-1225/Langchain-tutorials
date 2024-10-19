import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

import bs4
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory
from operator import itemgetter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain


loader = WebBaseLoader(web_paths =("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                        bs_kwargs= dict(
                            parse_only =  bs4.SoupStrainer(
                                class_ = ("post-content", 'post-title','post-header')
                            )
                        )        
                       )

docs =  loader.load()


text_splitter =  RecursiveCharacterTextSplitter(chunk_size = 1000, 
                                                chunk_overlap= 200,
                                                )
final_docs =  text_splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

vectorstore =  Chroma.from_documents(documents=  final_docs, embedding = embeddings, persist_directory=os.getcwd() + "/vectorstore"  )


retriever =  vectorstore.as_retriever()

llm = ChatGroq(model="llama3-8b-8192", groq_api_key = groq_api_key)

system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ('system','system_prompt'),
        MessagesPlaceholder("chat_history"),
        ('human','{input}')
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)

final_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", final_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store={}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_Rag_Chain = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key= 'input',
    history_messages_key= 'chat_history',
    output_messages_key= 'answer'
    )


st.title("Basic version of RAG-based Q&A Chatbot")
user_question = st.text_input("Please enter your question:") 

if user_question:
    response = conversational_Rag_Chain.invoke({
        'input': user_question
    },
    config = {
        'configurable':{'session_id':'ABC123'}
    }
    )
    st.write(f"Response to the question you asked is : {response['answer']}")
