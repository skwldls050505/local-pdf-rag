import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# -----------------------------
# 기본 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="Linux RAG Tutor",
    page_icon="🐧",
    layout="wide",
)


# -----------------------------
# CSS
# -----------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 950px;
            padding-top: 2rem;
        }

        .title-box {
            padding: 24px 28px;
            border-radius: 20px;
            background: #f5f7fa;
            margin-bottom: 20px;
        }

        .title-box h1 {
            margin: 0;
            font-size: 32px;
        }

        .title-box p {
            margin-top: 10px;
            color: #666;
        }

        .source-box {
            padding: 12px;
            background: #f7f7f7;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# 모델 로드
# Streamlit 재실행마다 모델을 다시 로드하지 않도록 cache 사용
# -----------------------------
@st.cache_resource
def load_rag():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="linux_pdf_clean",
    )

    llm = ChatOllama(
        model="qwen3:1.7b",
        temperature=0,
    )

    return vectorstore, llm


vectorstore, llm = load_rag()


# -----------------------------
# 제목
# -----------------------------
st.markdown(
    """
    <div class="title-box">
        <h1>🐧 Linux RAG Tutor</h1>
        <p>
        리눅스 강의자료를 기반으로 질문에 답변하는
        RAG 학습 도우미입니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# 채팅 기록
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# 기존 채팅 출력
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# 사용자 질문
# -----------------------------
question = st.chat_input(
    "예: pwd 명령어는 무엇을 하는 명령어야?"
)


if question:

    # 사용자 질문 화면 출력
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # -----------------------------
    # RAG 검색
    # -----------------------------
    docs = vectorstore.similarity_search(
        question,
        k=3,
    )


    # context 생성
    context_parts = []

    for doc in docs:

        page = doc.metadata.get(
            "page",
            "알 수 없음"
        )

        filename = doc.metadata.get(
            "filename",
            "리눅스 강의자료"
        )

        context_parts.append(
            f"""
[파일: {filename}]
[PDF {page}페이지]

{doc.page_content}
"""
        )


    context = "\n\n".join(context_parts)


    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = f"""
너는 리눅스 프로그래밍 강의 조교다.

아래 제공된 PDF 내용만을 바탕으로
사용자의 질문에 답해라.

규칙:

1. PDF 내용을 근거로 답한다.
2. PDF에 없는 내용을 임의로 만들어내지 않는다.
3. 초보자도 이해할 수 있도록 쉽게 설명한다.
4. 필요하면 간단한 명령어 예제를 보여준다.
5. 답변 마지막에 참고한 PDF 페이지를 표시한다.
6. PDF만으로 판단할 수 없다면
   "제공된 PDF 내용만으로는 확인하기 어렵습니다."
   라고 답한다.

[PDF 내용]

{context}


[사용자 질문]

{question}
"""


    # -----------------------------
    # LLM 답변
    # -----------------------------
    with st.chat_message("assistant"):

        with st.spinner("강의자료를 찾고 답변을 만들고 있어요..."):

            response = llm.invoke(prompt)

            answer = response.content

        st.markdown(answer)


        # -----------------------------
        # 검색된 근거 문서 표시
        # -----------------------------
        with st.expander("📚 참고한 PDF 내용 보기"):

            for i, doc in enumerate(docs, start=1):

                page = doc.metadata.get(
                    "page",
                    "알 수 없음"
                )

                filename = doc.metadata.get(
                    "filename",
                    "리눅스 강의자료"
                )

                st.markdown(
                    f"""
### {i}. {filename}

**PDF {page}페이지**

{doc.page_content[:1000]}
"""
                )

                st.divider()


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )