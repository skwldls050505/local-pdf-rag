
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama



# 1. Embedding 모델
# ingest.py에서 사용한 모델과 반드시 같아야 함
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Embedding 모델 준비 완료")


# 2. 기존 Chroma DB 불러오기
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="linux_pdf_clean",
)

print("Chroma DB 불러오기 완료")


# 3. 답변을 만들어줄 LLM
llm = ChatOllama(
    model="qwen3:1.7b",
    temperature=0,
)

print("LLM 준비 완료")


# 4. 반복해서 질문 받기
while True:

    question = input("\nPDF에 질문하세요 (종료: exit): ")

    if question.lower() in ["exit", "quit"]:
        print("종료합니다.")
        break

    # 5. 질문과 관련된 PDF Chunk 검색
    docs = vectorstore.similarity_search(
        question,
        k=3,
    )

    # 6. 검색된 Chunk를 하나의 Context로 합치기
    context_parts = []

    for doc in docs:
        page = doc.metadata.get("page", "알 수 없음")
        filename = doc.metadata.get("filename", "알 수 없음")

        context_parts.append(
            f"[파일: {filename} / PDF {page}페이지]\n"
            f"{doc.page_content}"
        )
        context = "\n\n".join(context_parts)


    # 7. LLM에게 전달할 Prompt
    prompt = f"""
당신은 리눅스 프로그래밍 강의자료를 설명해주는 조교입니다.

아래에 제공된 PDF 내용을 바탕으로 사용자의 질문에 답하세요.

규칙:
- PDF 내용에 근거해서 답하세요.
- PDF에 없는 내용을 함부로 만들어내지 마세요.
- 이해하기 쉽게 설명하세요.
- 필요하면 간단한 예제를 들어도 됩니다.
- 답변 마지막에는 참고한 PDF 페이지를 표시하세요.
- PDF 내용만으로 답하기 어렵다면 "제공된 PDF 내용만으로는 확인하기 어렵습니다."라고 말하세요.

[PDF 내용]
{context}

[사용자 질문]
{question}
"""


    # 8. LLM에게 질문
    response = llm.invoke(prompt)


    # 9. 답변 출력
    print("\n===== 답변 =====")
    print(response.content)


    # 10. 실제 검색된 페이지도 확인
    pages = []

    for doc in docs:
        page = doc.metadata.get("page", "알 수 없음")

        if page not in pages:
            pages.append(page)

    print("\n검색된 PDF 페이지:", pages)