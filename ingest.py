from pathlib import Path
import pymupdf4llm
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# LangChain에서는 Document 객체로 문서를 다루고, 
# text splitter를 이용해 긴 문서를 여러 조각으로 나누는 방식이 일반적인 RAG 전처리 흐름

def clean_text(text):
    # HTML 주석 제거
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

    # <br>, <u> 등 HTML 태그 제거
    text = re.sub(r"<[^>]+>", " ", text)

    # 이상한 글머리표 제거
    text = text.replace("", " ")
    text = text.replace("", " ")

    # 과도한 공백 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# 1. PDF 파일 찾기
pdf_files = list(Path("data").glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("data 폴더에 PDF 파일이 없습니다.")


documents = []


# 2. PDF 읽기
for pdf_path in pdf_files:
    print(f"\n불러오는 PDF: {pdf_path}")

    pages = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        show_progress=True,
        force_ocr=True,
        ocr_language="eng+kor",
    )

    for page in pages:
        text = clean_text(page["text"])

        if not text:
            continue

        page_number = page["metadata"]["page_number"]

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "filename": pdf_path.name,
                    "page": page_number,
                },
            )
        )
print(f"PDF 페이지 수: {len(pages)}")



# 3. LangChain Document 형식으로 변환
documents = []

for page in pages:
    text = clean_text(page["text"])

    if not text:
        continue

    page_number = page["metadata"]["page_number"]

    documents.append(
        Document(
            page_content=text,
            metadata={
                "source": str(pdf_path),
                "page": page_number,
            },
        )
    )


# 추출 상태 확인
print("\n===== 추출 결과 확인 =====")

for doc in documents[:5]:
    print(f"\n[PDF {doc.metadata['page']}페이지]")
    print(doc.page_content[:500])



# 4. Chunk로 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100, # 문장이 Chunk 경계에서 잘리는 문제를 줄이기 위해서
)

chunks = text_splitter.split_documents(documents)

print(f"\n생성된 Chunk 개수: {len(chunks)}")


# 5. 첫 번째 Chunk 확인
# print("\n===== 첫 번째 Chunk =====")
# print(chunks[0].page_content)

# print("\n===== Metadata =====")
# print(chunks[0].metadata)

# 6. Embedding 생성

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Embedding 모델 준비 완료")

# 7. Chroma Vector DB에 저장
# 여기 부분이 chunk -> embedding -> vector DB 저장 과정
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    persist_directory = "./chroma_db",
    collection_name = "linux_pdf_clean",

)

print("Chroma Vector DB 에 저장 완료 ! ! ")

