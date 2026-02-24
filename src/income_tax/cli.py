from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from pathlib import Path
from .utils import Utils


class CLI:

    @staticmethod
    def test_utils():
        """
        Utils.show_sim("king", "왕", client_type="openai")
        Utils.show_sim("king", "왕", client_type="upstage")
        """
        """
        # curl -fsSL https://ollama.com/install.sh | sh
        # ollama pull llama3
        # ollama list
        # sudo systemctl stop ollama
        #
        #
        # sudo systemctl start ollama
        Utils.msg_content(ChatOllama(model="llama3"), "한국의 마지막 왕은 누구였나요?")
        # sudo systemctl stop ollama
        load_dotenv()
        Utils.msg_content(ChatOpenAI(), "한국의 마지막 왕은 누구였나요?")
        Utils.msg_content(ChatUpstage(), "한국의 마지막 왕은 누구였나요?")
        """
        return

    @staticmethod
    def main():
        load_dotenv()
        persist_dir = Path.cwd() / "src" / "income_tax" / ".chroma"
        doc_path = Path.cwd() / "src" / "income_tax" / "assets" / "tax.docx"
        embedding = UpstageEmbeddings(
            model="solar-embedding-1-large-passage",
        )
        if persist_dir.exists():
            database = Chroma(
                collection_name="chroma-tax",
                persist_directory=str(persist_dir),
                embedding_function=embedding,
            )
        else:
            loader = Docx2txtLoader(doc_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
            )
            document_list = loader.load_and_split(text_splitter=text_splitter)
            database = Chroma.from_documents(
                documents=document_list,
                embedding=embedding,
                collection_name="chroma-tax",
                persist_directory=str(persist_dir),
            )
        llm = ChatUpstage(model="solar-1-mini-chat")
        prompt = ChatPromptTemplate.from_template(
            """
            당신은 최고의 한국 소득세 전문가입니다.

            반드시 아래 참고 문서에 근거해서만 답변하세요.
            참고 문서에 없는 내용은 절대 추측하지 마세요.
            모르면 모른다고 답하세요.
            
            참고 문서:
            {context}

            질문:
            {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(
            database.as_retriever(search_kwargs={"k": 3}), document_chain
        )
        query = "연봉 5천만원인 직장인의 소득세는 얼마인가요?"
        result = retrieval_chain.invoke({"input": query})
        print(f"{llm.__class__.__name__}:\n{result['answer']}\n")
