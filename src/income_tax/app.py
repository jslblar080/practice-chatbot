"""
streamlit run src/income_tax/app.py
"""

import streamlit as st


class App:

    def __init__(self) -> None:
        st.set_page_config(page_title="소득세 챗봇", page_icon="📄")
        st.title("📄 소득세 챗봇")
        st.caption("소득세와 관련된 모든 것을 답해드립니다!")
        if "message_list" not in st.session_state:
            st.session_state.message_list = []
        for message in st.session_state.message_list:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        if user_question := st.chat_input(
            placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"
        ):
            with st.chat_message("user"):
                st.write(user_question)
            st.session_state.message_list.append(
                {"role": "user", "content": user_question}
            )
            with st.spinner("답변을 생성하는 중입니다."):
                with st.chat_message("ai"):
                    st.write("여기는 AI 메시지")
                st.session_state.message_list.append(
                    {"role": "ai", "content": "여기는 AI 메시지"}
                )


if __name__ == "__main__":
    App()
