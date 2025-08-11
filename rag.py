import os
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import streamlit as st
from langchain_chroma import Chroma
from langchain.schema import AIMessage
from huggingface_hub import snapshot_download
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
    # 1. DB 로드 함수를 실행하여 DB 파일이 저장된 경로를 가져옵니다.
    db_path = "./data/namuwiki_db_hf"
    if not os.path.exists(db_path):
        snapshot_download(
            repo_id="SoccerData/namuwiki_db",
            repo_type="dataset",
            local_dir=db_path,
            local_dir_use_symlinks=False
        )

    # 2. 임베딩 함수를 정의합니다.
    embedding_function = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function
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

st.header("축구 Q&A 챗봇 ⚽")

# st.spinner를 사용해 전체 초기화 과정 중에 메시지를 표시
with st.spinner("챗봇을 초기화하는 중입니다. DB 다운로드 및 모델 로딩..."):
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
            context_docs = response['context']

            st.write(answer)

            # # 3. st.expander를 사용해 참조한 문서(RAG 검색 결과)를 깔끔하게 표시합니다.
            # if context_docs: # 검색된 문서가 있을 경우에만 표시
            #     with st.expander("참조한 문서 보기 (RAG 검색 결과)"):
            #         for doc in context_docs:
            #             st.markdown(f"**출처: `{doc.metadata.get('source', '정보 없음')}`**")
            #             st.markdown(doc.page_content)
            #             st.divider()

            
