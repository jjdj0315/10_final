# app.py
import streamlit as st
from dotenv import load_dotenv
from utils.session import session_control
from utils.uuid import random_uuid
from utils.print_messages import print_messages

# RAG 앱 관련 임포트
from utils.create_dir import create_dir
from utils.creat_compression_retriever import creat_compression_retriever
from utils.upload import upload_file
from utils.node import create_app
from utils.add_message import add_message  # add_message 함수가 필요
from utils.handler import stream_handler

# LangChain 메시지 타입 임포트 (LangGraph 입력용)
from langchain_core.messages import HumanMessage, AIMessage, ChatMessage, ToolMessage


load_dotenv()
session_control()
create_dir()  # 필요시 디렉토리 생성

st.title("LOCAL RAG LLM")
st.markdown("온프라미스 RAG LLM입니다., 멀티턴대화를 지원합니다.")

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    st.markdown("# **made by JDJ**")
    selected_loader = st.radio(
        "로더 선택", ["docling", "PDFPlumber", "정대진"], index=0
    )
    file = st.file_uploader("PDF 파일 업로드", type=["pdf"])
    apply_btn = st.button("설정 완료", type="primary")

# 초기화 버튼
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["thread_id"] = random_uuid()
    # RAG 관련 세션 상태도 초기화
    if "app" in st.session_state:
        del st.session_state["app"]
    if "compression_retriever" in st.session_state:
        del st.session_state["compression_retriever"]
    st.rerun()  # 초기화 후 화면 새로고침

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요")

# 경고메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 이전 대화기록출력
print_messages()

# 설정 버튼이 눌리면
if apply_btn:
    if file:
        with st.spinner("파일 처리 및 RAG 설정 중..."):
            FILE_PATH = upload_file(file)
            st.session_state["compression_retriever"] = creat_compression_retriever(
                FILE_PATH, selected_loader
            )
            st.session_state["app"] = create_app()  # LangGraph 앱 생성
            st.session_state["thread_id"] = random_uuid()  # 새로운 스레드 ID 생성
            st.success("✅ RAG 설정 완료!")
    else:
        st.error("PDF 파일을 먼저 업로드해주세요.")
    st.rerun()  # 설정 적용 후 화면 새로고침


# 만약에 사용자 입력이 들어오면
if user_input:
    # 1. 사용자 메시지 화면에 즉시 출력
    st.chat_message("user").write(user_input)

    # 2. LangGraph에 전달할 전체 메시지 기록 준비
    # st.session_state["messages"]에는 utils.dataclass.ChatMessageWithType 객체가 저장되어 있습니다.
    # LangGraph의 `messages` 필드는 langchain_core.messages.BaseMessage 타입을 기대합니다.
    langgraph_messages_history = []
    for msg_with_type in st.session_state["messages"]:
        # ChatMessageWithType 객체에서 chat_message 속성을 추출하여 LangGraph에 전달
        langgraph_messages_history.append(msg_with_type.chat_message)

    # 현재 사용자 질문을 LangGraph의 messages 입력에 추가 (HumanMessage 타입으로)
    langgraph_messages_history.append(HumanMessage(content=user_input))

    # config 설정 (thread_id 포함)
    if "app" in st.session_state and st.session_state["app"] is not None:
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

        with st.chat_message("assistant"):
            # 각 UI 요소를 위한 placeholder를 명시적으로 생성
            node_status_placeholder = st.empty()  # 노드 상태 메시지용
            retrieved_docs_expander_placeholder = st.empty()  # 검색된 문서 expander용
            thinking_placeholder = st.empty()  # 추론 과정 스트리밍용
            answer_placeholder = st.empty()  # 최종 답변 스트리밍용

            # LangGraph inputs 설정
            inputs = {
                "query": user_input,  # 현재 쿼리는 분류/검색을 위해 별도로 전달
                "messages": langgraph_messages_history,  # <--- 여기가 핵심 변경점!
                "documents": [],  # 초기화
                "thinking": "",  # 초기화
                "answer": "",  # 초기화
                "mode": "",  # 초기화
            }

            try:
                # LangGraph 실행 및 스트리밍 제너레이터 얻기
                langgraph_stream_generator = st.session_state["app"].stream(
                    inputs,
                    config,  # 'st.session_state["config"]' 대신 직접 `config` 변수 사용
                    stream_mode="updates",
                )

                # stream_handler 호출하여 UI 업데이트 및 최종 결과 받기
                # (stream_handler가 이제 `invoke()`값을 스트리밍처럼 처리)
                retrieved_docs, final_answer, final_thinking = stream_handler(
                    node_status_placeholder,
                    retrieved_docs_expander_placeholder,
                    thinking_placeholder,
                    answer_placeholder,
                    langgraph_stream_generator,
                )

                # 3. 스트리밍이 완료된 후, 최종 결과들을 session_state["messages"]에 추가
                # 현재 사용자 메시지를 먼저 session_state에 추가 (화면 출력은 이미 위에서 함)
                add_message("user", user_input)

                # 검색 결과가 있다면 session_state에 추가
                if retrieved_docs:
                    # format_search_result 함수가 List[Document]를 받도록 수정되었으므로 그대로 전달
                    add_message(
                        "assistant",
                        retrieved_docs,
                        "tool_result",  # 메시지 타입
                        "문서 검색 결과",  # 툴 이름 (expander 헤더용)
                    )

                # 추론 과정이 있다면 session_state에 추가
                if final_thinking:
                    add_message(
                        "assistant",
                        f"**🧠 추론 과정:**\n{final_thinking}",  # 추론 과정은 일반 텍스트로 저장
                        "text",
                        "추론 과정",  # 식별자 (필요시)
                    )

                # 최종 답변이 있다면 session_state에 추가
                if final_answer:
                    add_message("assistant", final_answer, "text")

            except Exception as e:
                error_message = f"오류 발생: {e}"
                st.error(f"LangGraph 실행 중 오류 발생: {e}")
                add_message(
                    "assistant", error_message, "text"
                )  # 오류 메시지도 대화에 추가

    else:
        warning_msg.warning("PDF 파일을 업로드하고 설정을 완료해주세요.")

# 세션 상태 디버깅용 expander
with st.expander("🔍 세션 상태 보기"):
    for key, value in st.session_state.items():
        if key == "messages":
            st.write(f"{key}:")
            for msg in value:
                # 메시지 내용을 너무 길게 출력하지 않도록 자르기
                st.write(
                    f"  - {msg.chat_message.role}: {str(msg.chat_message.content)[:50]}..."
                )
        else:
            st.write(f"{key}: {value}")
