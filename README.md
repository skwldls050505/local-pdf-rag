# 📚 Local PDF RAG

PDF 강의자료를 기반으로 질문에 답변하는 로컬 RAG(Retrieval-Augmented Generation) 실습 프로젝트입니다.

PDF의 내용을 추출하고 Chunk 단위로 분할한 뒤 Embedding을 생성하여 Chroma Vector DB에 저장합니다.  
사용자가 질문하면 질문과 의미적으로 유사한 문서를 검색하고, 검색된 내용을 Local LLM에 전달하여 PDF 기반 답변을 생성합니다.

---

## 프로젝트 목표

RAG의 전체 흐름을 직접 구현하면서 아래 과정을 이해하는 것을 목표로 진행했습니다.

PDF
→ Text Extraction / OCR
→ Chunking
→ Embedding
→ Vector DB
→ Retrieval
→ LLM
→ Answer

단순히 LangChain의 완성된 RAG Chain을 사용하는 대신, 각 단계를 직접 연결하면서 내부 동작을 확인했습니다.

---

## RAG Architecture

```text
PDF 강의자료
      ↓
PyMuPDF4LLM + Tesseract OCR
      ↓
텍스트 추출 및 전처리
      ↓
RecursiveCharacterTextSplitter
      ↓
Chunk 생성
      ↓
HuggingFace Embedding
      ↓
Chroma Vector DB
      ↓
사용자 질문
      ↓
Question Embedding
      ↓
Similarity Search
      ↓
관련 PDF Chunk 검색
      ↓
Ollama Local LLM
      ↓
PDF 내용을 기반으로 답변 생성
      ↓
Streamlit UI
