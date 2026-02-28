"""
streamlit run src/income_tax/app.py
"""

import streamlit as st

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.base import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pathlib import Path

# my own module
from income_tax.utils import Utils


class App(metaclass=Utils.SingletonMeta):

    _rewrite_chain: RunnableSerializable
    _retrieval_chain: Runnable

    def __init__(self) -> None:
        load_dotenv()
        doc_path = (
            Path.cwd() / "src" / "income_tax" / "assets" / "tax_with_markdown.docx"
        )
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-large",
        )
        # database = Utils.build_chroma_db(
        #     doc_path, embedding, (Path.cwd() / "src" / "income_tax" / ".chroma")
        # )
        database = Utils.build_pinecone_db(doc_path, embedding)
        llm = ChatOpenAI(model="gpt-5.2")
        rewrite_prompt = ChatPromptTemplate.from_template(
            """
            당신은 언어 전문가입니다.

            아래 규칙을 참고하여 질문을 변환하세요.

            [규칙]
            - 사람을 나타내는 표현(직장인, 개인, 사람 등)은 필요한 경우에만 "거주자"로 변환하세요.
            - 변환이 필요 없다면 원문을 그대로 출력하세요.
            - 다른 설명은 하지 말고 변환된 질문만 출력하세요.

            [질문]
            {input}
            """
        )
        self._rewrite_chain = rewrite_prompt | llm | StrOutputParser()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 한국 소득세 전문가입니다.\n\n"
                    "참고 문서에 근거하여 답변하세요.\n"
                    "참고 문서에 없는 사실은 생성하지 마세요.\n\n"
                    "계산이 필요한 경우:\n"
                    "1. 모든 구간을 나누어 계산합니다.\n"
                    "2. 각 구간별 계산 결과를 제시합니다.\n"
                    "3. 모든 구간을 검토하여 최종 결과를 제시합니다.\n"
                    "4. 단일 구간 공식만 적용하여 계산을 종료하지 마세요.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "[참고 문서]\n{context}\n\n" "[질문]\n{input}"),
            ]
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "이전 대화 내용을 참고하여 최신 질문을 "
                    "검색용 독립 질문으로 재작성하세요.\n"
                    "판단에 필요한 모든 명시적 정보를 포함하세요.\n"
                    "추론하거나 새로운 정보를 추가하지 마세요.\n"
                    "재작성된 질문만 출력하세요.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm,
            database.as_retriever(search_kwargs={"k": 4}),
            contextualize_q_prompt,
        )
        self._retrieval_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain,
        )

    def run(self):
        st.set_page_config(page_title="소득세 챗봇", page_icon="📄")
        st.title("📄 소득세 챗봇")
        st.caption("소득세와 관련된 모든 것을 답해드립니다!")
        if "message_list" not in st.session_state:
            st.session_state.message_list = []
        for msg in st.session_state.message_list:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if user_query := st.chat_input(
            placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!",
        ):
            with st.chat_message("user"):
                st.write(user_query)
            chat_history = []
            for msg in st.session_state.message_list:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            normalized_query = self._rewrite_chain.invoke({"input": user_query})
            st.session_state.message_list.append(
                {"role": "user", "content": user_query}
            )
            with st.status("잠시만 기다려주세요...") as status:
                result = self._retrieval_chain.invoke(
                    {
                        "input": normalized_query,
                        "chat_history": chat_history,
                    }
                )
                status.update(
                    label=f"챗봇이 이해한 메시지:\n{normalized_query}", state="complete"
                )
            with st.chat_message("ai"):
                st.write(result["answer"])
            st.session_state.message_list.append(
                {"role": "ai", "content": result["answer"]}
            )


if __name__ == "__main__":
    app = App()
    app.run()
