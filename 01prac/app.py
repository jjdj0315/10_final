import streamlit as st


st.title("local langgraph agent")
st.markdown("local langgraph agent")

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요")

# 경고메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면
if user_input:
    # 1. 사용자 메시지 화면에 즉시 출력
    st.chat_message("user").write(user_input)
