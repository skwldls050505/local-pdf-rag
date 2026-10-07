from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Embedding 모델 불러오기
# ingest.py에서 사용한 모델과 반드시 같아야 함!
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Embedding 모델 준비 완료")


# 2. 저장해둔 Chroma DB 불러오기
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="linux_pdf_clean"
)

print("Chroma DB 불러오기 완료")


# 3. 질문 입력
question = input("\nPDF에 질문해보세요: ")


# 4. 질문과 비슷한 Chunk 검색
results = vectorstore.similarity_search(
    question,
    k=5 # 질문과 의미가 가장 비슷한 Chunk를 3개 가져와! 
)


# 5. 검색 결과 출력
print("\n===== 검색 결과 =====")

for i, doc in enumerate(results, start=1):
    page = doc.metadata.get("page", "알 수 없음")
    filename = doc.metadata.get("filename", "알 수 없음")

    print(f"\n[{i}번 결과]")
    print(f"파일: {filename}")
    print(f"페이지: {page}")
    print(doc.page_content)
    print("-" * 60)