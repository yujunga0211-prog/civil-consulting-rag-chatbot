import os
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================
# 1. Streamlit 기본 설정
# =========================================

st.set_page_config(
    page_title="공공 민원 상담원 응대 지원 RAG",
    page_icon="📘",
    layout="wide",
)

BASE_DIR = Path(__file__).parent.resolve()

ZIP_PATH = BASE_DIR / "civil_consulting_qa_all.zip"
FAISS_DIR = BASE_DIR / "civil_streamlit_faiss_db_all"

MAX_ROWS = None


# =========================================
# 2. API Key 설정
# =========================================

def set_openai_api_key():
    """
    로컬 실행:
    - app.py와 같은 폴더의 .env 파일 사용 가능
    - 또는 시스템 환경변수 OPENAI_API_KEY 사용 가능

    Streamlit Cloud 배포:
    - Manage app > Settings > Secrets에 아래 형식으로 등록
      OPENAI_API_KEY = "sk-..."
    """

    load_dotenv(BASE_DIR / ".env")

    api_key = None

    # 1순위: Streamlit Cloud Secrets
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

    # 2순위: 로컬 .env 또는 시스템 환경변수
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "OPENAI_API_KEY가 설정되어 있지 않습니다.\n\n"
            "Streamlit Cloud에서는 Manage app > Settings > Secrets에 다음 형식으로 등록하세요.\n\n"
            'OPENAI_API_KEY = "sk-..."'
        )
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key


set_openai_api_key()


# =========================================
# 3. CSS
# =========================================

st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 50%, #f0fdf4 100%);
    }

    .hero-card {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid rgba(226, 232, 240, 0.95);
        border-radius: 28px;
        padding: 38px;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
        margin-bottom: 22px;
    }

    .badge {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 16px;
    }

    .main-title {
        font-size: 40px;
        line-height: 1.25;
        letter-spacing: -1.2px;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 14px;
    }

    .subtitle {
        color: #475569;
        font-size: 17px;
        line-height: 1.8;
    }

    .info-card {
        background: #0f172a;
        color: white;
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
        margin-bottom: 18px;
    }

    .info-card h3 {
        margin-top: 0;
        color: white;
    }

    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: 16px;
        padding: 18px;
        margin-top: 12px;
        line-height: 1.65;
        color: #334155;
        font-size: 14px;
    }

    .answer-card {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px;
        min-height: 360px;
        line-height: 1.7;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
        white-space: pre-wrap;
    }

    .rag-card {
        border-top: 6px solid #2563eb;
    }

    .llm-card {
        border-top: 6px solid #f97316;
    }

    .small-note {
        color: #64748b;
        font-size: 13px;
        line-height: 1.6;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================
# 4. 데이터 로드
# =========================================

@st.cache_data(show_spinner=False)
def load_data():
    if not ZIP_PATH.exists():
        st.error(
            f"데이터 압축 파일을 찾을 수 없습니다: {ZIP_PATH.name}\n\n"
            "GitHub에 civil_consulting_qa_all.zip 파일이 app.py와 같은 위치에 있는지 확인하세요."
        )
        st.stop()

    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            csv_files = [
                name for name in z.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_files:
                st.error("zip 파일 안에 CSV 파일이 없습니다.")
                st.stop()

            csv_name = csv_files[0]

            with z.open(csv_name) as f:
                if MAX_ROWS is not None:
                    df = pd.read_csv(f, nrows=MAX_ROWS)
                else:
                    df = pd.read_csv(f)

    except Exception as e:
        st.error(f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    required_cols = [
        "data_group",
        "source",
        "consulting_category",
        "question",
        "answer",
        "context",
        "consulting_content",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["main_context"] = df.apply(
        lambda row: row["context"]
        if str(row["context"]).strip()
        else row["consulting_content"],
        axis=1,
    )

    df["rag_text"] = (
        "[기관구분] " + df["data_group"].astype(str) + "\n"
        + "[기관명] " + df["source"].astype(str) + "\n"
        + "[상담분야] " + df["consulting_category"].astype(str) + "\n"
        + "[상담 질문]\n" + df["question"].astype(str) + "\n"
        + "[상담 답변]\n" + df["answer"].astype(str) + "\n"
        + "[상담 원문]\n" + df["main_context"].astype(str)
    )

    df = df[df["rag_text"].astype(str).str.len() >= 50]
    df = df.reset_index(drop=True)

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)

    return df


# =========================================
# 5. FAISS 벡터DB 생성/로드
# =========================================

@st.cache_resource(show_spinner=False)
def create_or_load_vectorstore():
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    if FAISS_DIR.exists():
        try:
            vectorstore = FAISS.load_local(
                str(FAISS_DIR),
                embedding,
                allow_dangerous_deserialization=True,
            )
            return vectorstore
        except Exception as e:
            st.warning(
                f"기존 FAISS 벡터DB를 불러오는 중 오류가 발생했습니다: {e}\n\n"
                "벡터DB를 다시 생성합니다."
            )

    df = load_data()

    documents = []

    for _, row in df.iterrows():
        metadata = {
            "data_group": str(row.get("data_group", "")),
            "source": str(row.get("source", "")),
            "consulting_category": str(row.get("consulting_category", "")),
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
        }

        documents.append(
            Document(
                page_content=str(row["rag_text"]),
                metadata=metadata,
            )
        )

    if not documents:
        st.error("벡터DB를 만들 문서가 없습니다. CSV 데이터 내용을 확인하세요.")
        st.stop()

    try:
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embedding,
        )

        vectorstore.save_local(str(FAISS_DIR))

        return vectorstore

    except Exception as e:
        st.error(
            f"FAISS 벡터DB 생성 중 오류가 발생했습니다: {e}\n\n"
            "가능한 원인:\n"
            "1. OPENAI_API_KEY가 잘못되었거나 결제/사용량 문제가 있음\n"
            "2. 데이터 수가 너무 많아 Streamlit Cloud에서 처리 시간이 초과됨\n"
            "3. faiss-cpu 또는 langchain 관련 패키지 설치 문제\n\n"
        
        )
        st.stop()


# =========================================
# 6. 검색어 확장
# =========================================

def expand_query(query):
    synonym_map = {
        "민증": "주민등록증 신분증 주민등록증 발급 주민센터",
        "신분증": "주민등록증 신분증 발급 주민센터",
        "주민등록증": "주민등록증 신분증 발급 주민센터",

        "대출": "대출 신용보증재단 보증재단 지역신용보증재단 소상공인 개인사업자 정책자금",
        "보증재단": "신용보증재단 지역신용보증재단 보증서 대출 소상공인",
        "신용보증재단": "신용보증재단 지역신용보증재단 보증서 대출 소상공인",
        "소상공인": "소상공인 개인사업자 창업 대출 정책자금 지원",
        "개인사업자": "개인사업자 소상공인 사업자등록 대출 신용보증재단",
        "정책자금": "정책자금 소상공인 대출 창업자금 신용보증재단",

        "퇴직": "퇴직 임금 수당 급여 근로기준법 고용노동부",
        "임금": "임금 수당 퇴직 급여 근로기준법 고용노동부",
        "수당": "임금 수당 퇴직 급여 근로기준법 고용노동부",
        "급여": "급여 임금 수당 퇴직 근로기준법",

        "장애인기업": "장애인기업확인서 장애인기업 확인서 발급 창업 중소벤처기업부",
        "확인서": "확인서 발급 서류 절차 신청 장애인기업확인서",

        "자동차": "자동차 등록 이전등록 자동차등록원부 사용본거지",
        "부동산": "부동산 공인중개사 계약서 직거래 행정사",
    }

    expanded = query

    for key, value in synonym_map.items():
        if key in query:
            expanded += " " + value

    return expanded


# =========================================
# 7. RAG 검색/생성
# =========================================

def retrieve_documents(vectorstore, query, k=4):
    expanded_query = expand_query(query)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
        },
    )

    docs = retriever.invoke(expanded_query)
    return docs


def trim_text(text, max_chars=3500):
    text = str(text)

    if len(text) > max_chars:
        return text[:max_chars] + "\n...[일부 생략]"

    return text


def format_docs(docs):
    return "\n\n---\n\n".join(
        [trim_text(doc.page_content, 3000) for doc in docs]
    )


def generate_answer(query, docs):
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_template(
        """
당신은 공공 민원 상담원 교육을 돕는 AI 코치입니다.

이 서비스의 대상은 일반 시민이 아니라 신입 공공 민원 상담원입니다.
따라서 최신 민원 절차를 확정적으로 안내하기보다,
과거 상담 사례를 바탕으로 상담원이 참고할 수 있는 응대 방식과 말투를 알려주는 것이 목적입니다.

중요 규칙:
- 실제 기관명으로 자신을 소개하지 마세요.
- 예를 들어 "중소벤처기업부입니다", "고용노동부입니다"라고 말하지 마세요.
- 대신 "과거 상담 사례를 참고하면", "검색된 상담 사례에서는"처럼 표현하세요.
- 검색된 사례에 없는 구체적인 서류, 기간, 장소는 임의로 만들어 말하지 마세요.
- 검색된 사례가 질문과 직접 관련이 약하면, "검색된 사례만으로는 직접적인 답변을 만들기 어렵다"고 말하세요.
- 답변은 민원 정답 제공이 아니라 상담 응대 방식 학습용으로 작성하세요.
- 답변은 한국어로 작성하세요.

답변은 반드시 다음 형식을 따르세요.

## 1. 상담원 응대 예시
- 실제 상담원이 민원인에게 답변하듯이 공손한 문장으로 작성하세요.
- 단정적인 최신 정책 안내보다는 "과거 상담 사례를 참고하면", "관련 사례에서는"과 같은 표현을 사용하세요.

## 2. 답변 구조 분석
- 민원 요지 확인:
- 안내 또는 근거 제시:
- 추가 문의 안내:
- 마무리 표현:

## 3. 신입 상담원이 배울 점
- 공손한 표현:
- 민원 내용을 정리하는 방식:
- 근거를 제시하는 방식:
- 불확실한 내용에 대해 조심스럽게 안내하는 방식:

## 4. 참고한 유사 상담 사례 요약
- 검색된 사례의 핵심 내용을 짧게 정리하세요.

[유사 과거 상담 사례]
{context}

[상담원이 입력한 질문]
{question}
"""
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "context": context,
            "question": query,
        }
    )


def generate_llm_answer(query):
    """
    검색 문서 없이 LLM만으로 생성하는 비교용 답변.
    교수님 요구사항: UI에서 RAG 결과와 LLM 결과를 나란히 보여 주기 위한 함수.
    """
    prompt = ChatPromptTemplate.from_template(
        """
당신은 공공 민원 상담원 교육을 돕는 AI 코치입니다.

아래 질문에 대해 검색 문서 없이 일반 LLM 지식만으로 답변하세요.
이 답변은 RAG 기반 답변과 비교하기 위한 검증용 답변입니다.

중요 규칙:
- 실제 기관명으로 자신을 소개하지 마세요.
- 검색 근거가 없으므로 구체적인 서류, 기간, 법령, 신청 장소를 확정적으로 단정하지 마세요.
- 불확실한 내용은 "확인이 필요하다", "담당 기관 확인이 필요하다"처럼 조심스럽게 표현하세요.
- 답변은 신입 상담원이 참고할 수 있는 응대 예시 형식으로 작성하세요.
- 답변은 한국어로 작성하세요.

답변은 다음 형식을 따르세요.

## 일반 LLM 응대 예시
- 검색 문서 없이 생성한 일반적인 응대 예시를 작성하세요.

## 주의할 점
- 이 답변은 검색된 과거 상담 사례에 근거하지 않았으므로 실제 상담에서는 추가 확인이 필요하다는 점을 설명하세요.

[상담원이 입력한 질문]
{question}

[일반 LLM 답변]
"""
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"question": query})


# =========================================
# 8. Streamlit 화면
# =========================================

st.markdown(
    """
<div class="hero-card">
    <div class="badge">RAG 기반 상담 교육 보조 시스템</div>
    <div class="main-title">공공 민원 상담원<br>응대 지원 챗봇</div>
    <div class="subtitle">
        일반 시민에게 최신 민원 정답을 직접 제공하는 서비스가 아니라,<br>
        신입 공공 민원 상담원이 과거 상담 사례를 참고해 공손한 말투와 답변 구조를 학습할 수 있도록 돕는 서비스입니다.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 서비스 설정")
    search_k = st.slider("검색할 유사 상담 사례 수", 1, 6, 4)

    st.markdown("---")
    st.markdown("### 데이터 정보")

    df = load_data()
    st.metric("사용 데이터 수", f"{len(df):,}")
    st.metric("기관 수", f"{df['source'].nunique():,}")
    st.metric("상담 분야 수", f"{df['consulting_category'].nunique():,}")

    st.markdown("---")
    st.markdown(
        """
**추천 질문**
- 개인사업자가 신용보증재단 대출 가능 여부를 문의할 때 신입 상담원은 어떻게 응대하면 좋나요?
- 퇴직한 근로자가 임금이나 수당을 받을 수 있는지 문의할 때 신입 상담원은 어떤 구조로 답변하면 좋나요?
- 장애인기업확인서 발급 절차를 문의하는 민원인에게 어떻게 답변하면 좋나요?
- 부동산 직거래 계약서 작성 가능 여부를 문의하면 어떤 기준을 안내해야 하나요?
- 자동차 이전등록 관련 문의에서는 어떤 정보를 확인해야 하나요?
"""
    )

col1, col2 = st.columns([0.9, 1.1])

with col1:
    st.markdown(
        """
<div class="info-card">
    <h3>서비스 핵심 기능</h3>
    <p>· 유사한 과거 민원 상담 사례 검색</p>
    <p>· RAG 기반 답변과 일반 LLM 답변 비교 출력</p>
    <p>· 상담원 응대 예시 및 학습 포인트 제공</p>
    <p>· 검색 근거 사례를 함께 출력하여 RAG 구조 확인</p>
</div>
""",
        unsafe_allow_html=True,
    )

    default_question = (
        "개인사업자가 신용보증재단 대출 가능 여부를 문의할 때 "
        "신입 상담원은 어떻게 응대하면 좋나요?"
    )

    user_question = st.text_area(
        "상담 상황 또는 질문 입력",
        value=default_question,
        height=170,
    )

    run_button = st.button("질문하기", type="primary", use_container_width=True)

with col2:
    st.markdown("### 답변 비교")
    st.info("질문을 입력하면 RAG 기반 답변과 일반 LLM 답변이 나란히 표시됩니다.")


if run_button:
    if not user_question.strip():
        st.warning("질문을 입력하세요.")
        st.stop()

    with st.spinner(
        "공공 민원 데이터를 기반으로 RAG 답변과 일반 LLM 답변을 생성하는 중입니다. "
        "첫 실행은 벡터DB 생성 때문에 시간이 걸릴 수 있습니다..."
    ):
        vectorstore = create_or_load_vectorstore()
        docs = retrieve_documents(vectorstore, user_question, k=search_k)
        rag_answer = generate_answer(user_question, docs)
        llm_answer = generate_llm_answer(user_question)

    st.markdown("## 답변 비교")

    rag_col, llm_col = st.columns(2)

    with rag_col:
        st.markdown("### RAG 기반 답변")
        st.caption("검색된 과거 상담 사례를 근거로 생성한 답변")
        st.markdown(
            f"""
<div class="answer-card rag-card">
{rag_answer}
</div>
""",
            unsafe_allow_html=True,
        )

    with llm_col:
        st.markdown("### 일반 LLM 답변")
        st.caption("검색 문서 없이 LLM만으로 생성한 비교용 답변")
        st.markdown(
            f"""
<div class="answer-card llm-card">
{llm_answer}
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("### 검색된 유사 상담 사례")
    st.caption("RAG 기반 답변 생성에 실제로 사용된 검색 근거입니다.")

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        with st.expander(
            f"사례 {i} | {metadata.get('source', '')} / {metadata.get('consulting_category', '')}",
            expanded=(i == 1),
        ):
            st.markdown(
                f"""
<div class="source-card">
<b>기관구분:</b> {metadata.get("data_group", "")}<br>
<b>기관명:</b> {metadata.get("source", "")}<br>
<b>상담분야:</b> {metadata.get("consulting_category", "")}<br>
<b>기존 질문:</b> {metadata.get("question", "")}<br>
<b>기존 답변:</b> {metadata.get("answer", "")}<br>
</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown("**원문 일부**")
            st.write(trim_text(doc.page_content, 1200))

st.markdown(
    """
<div class="small-note">
AI Hub 공공 민원 상담 데이터를 활용한 신입 상담원 응대 학습용 RAG 프로토타입입니다.
</div>
""",
    unsafe_allow_html=True,
)