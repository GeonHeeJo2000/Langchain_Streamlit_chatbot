import os
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import streamlit as st
from langchain_chroma import Chroma
from langchain.schema import AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

#오픈AI API 키 설정
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

@st.cache_resource
def initialize_components():
    vectorstore = Chroma(
            persist_directory="./data/namuwiki_db",
            embedding_function=HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", model_kwargs={'device': 'cpu'})
        )

    retriever = vectorstore.as_retriever()
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. 대답은 한국어로 하고, 존댓말을 써줘.\
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini")
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

st.header("축구 Q&A 챗봇")
rag_chain = initialize_components()
chat_history = StreamlitChatMessageHistory(key="chat_messages")
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

if not chat_history.messages:
    chat_history.messages = [AIMessage(content="축구 대해 무엇이든 물어보세요!")]
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)
if prompt_message := st.chat_input("Your question"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke({"input": prompt_message}, config)
            answer = response['answer']
            st.write(answer)


