from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Literal

from .state import State
from .model import reasoning_llm, answer_llm


classify_llm_chain = ChatPromptTemplate.from_template(
    """
다음 사용자 질문이 외부 문서의 정보가 필요한지 판단하세요.

- 특정 문서, 보고서, 사용자 업로드 파일, 수치, 통계, 날짜 등 **문서 기반 정보가 필요한 경우**: 'retrieve'
- 일반 상식, 대화, 감정, 일상적인 질문처럼 **문서 없이도 대답 가능한 경우**: 'generate'

오직 아래 두 단어 중 하나만 출력하세요: retrieve 또는 generate  
그 외 문장은 절대 출력하지 마세요.

사용자 질문: {query}

판단:
    """
    | reasoning_llm
    | StrOutputParser
)


# 1. 질문 분류 함수 - 중요: 여기서는 상태를 업데이트하는 노드 함수
def classify_node(state: State):
    """질문을 분류하여 처리모드를 결저합니다."""
    query = state["query"]
    print("질문 분류 시작 : {query}")

    # 정의된 classify_llm_chain을 사용하야 질문 의도 분류
    classification = classify_llm_chain.invoke({"query": query})

    print(f"====질문 분류 완료 : {classification}====")

    # 분류 결과에 따라 mode상태를 설정합니다.
    if "retrieve" in classification.lower():
        mode = "retreive"
    else:
        mode = "generate"

    print("모드 결정 {mode}")
    return {"mode": mode}


# 추론 노드
def reasoning(state: State):
    """검색된 문서와 질문을 기반으로 추론 과정을 생성합니다."""
    query = state["query"]
    documents = state.get("documents", [])

    # 문서 내용을 추출하여 컨텍스트로 사용
    context = "\n\n".join([doc.page_context for doc in documents])

    reasoning_prompt = ChatPromptTemplate.from_template(
        """주어진 문서를 활용하여 사용자의 질문에 가장 적절한 답변을 작성해주세요.
        문서가 없다면 일반적인 지식으로 답변을 시도하세요.

        질문: {query}

        문서 내용:
        {context}


        상세 추론:"""
    )

    reasoning_chain = reasoning_prompt | reasoning_llm | StrOutputParser()

    print("추론 시작")

    thinking = reasoning_chain.invoke({"query": query, "context": context})
    print("===추론완료===")
    return {"thinking": thinking}


# 2. 문서 검색 노드 (Retriever)
def retrieve(state: State):
    """압축 리트리버를 사용하여 관련 문서를 검색합니다."""
    print("=====문서 검색 시작=====")
    query = state["query"]
    retriever = st.session_state["compression_retriever"]
    documents = retriever.invoke(query)
    print("=====문서 검색 완료=====")
    return {"documents": documents}


# 4. 답변 생성 노드 (Answer LLM)
def generate(state: State):
    """문서와 추론 과정을 기반으로 최종 답변을 생성합니다."""

    query = state["query"]
    thinking = state.get("thinking", "")  # get 메서드로 안전하게 접근
    documents = state.get("documents", [])  # get 메서드로 안전하게 접근

    # 문서 내용 추출
    context = "\n\n".join([doc.page_content for doc in documents])

    # 최종 답변 생성을 위한 프롬프트
    answer_prompt = ChatPromptTemplate.from_template(
        """사용자 질문에 한글로 답변하세요. 제공된 문서와 추론 과정이 있다면, 최대한 활용하세요.
        문서나 추론 과정이 부족하더라도 사용자 질문에 자연스럽게 답변하세요.

        질문: {query}

        추론 과정:
        {thinking}

        문서 내용:
        {context}

        답변:"""
    )

    answer_chain = answer_prompt | answer_llm | StrOutputParser()

    print("=====답변 생성 시작=====")
    # stream() 대신 invoke()를 사용하여 전체 결과 한 번에 받기
    answer = answer_chain.invoke(
        {"query": query, "context": context, "thinking": thinking}
    )
    print("=====답변 생성 완료=====")
    return {"answer": answer}


# 라우팅 함수 - 중요: 여기서는 다음 노드를 결정하는 라우팅 함수
def route_by_mode(state: State) -> Literal["retrieve", "generate"]:
    """분류 결과에 따라 다음 노드를 결정합니다."""
    print(f"=====라우팅: {state['mode']}=====")
    if state["mode"] == "retrieve":
        return "retrieve"
    elif state["mode"] == "generate":
        return "generate"
    raise ValueError("Invalid mode set in state")
