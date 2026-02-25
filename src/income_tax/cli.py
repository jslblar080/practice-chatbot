from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pathlib import Path
from .docx_image_to_markdown import DocxImageToMarkdown
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
    def convert_docx_img_to_md():
        load_dotenv()
        assets_dir = Path.cwd() / "src" / "income_tax" / "assets"
        converter = DocxImageToMarkdown()
        converter.convert(
            str(assets_dir / "tax.docx"), str(assets_dir / "tax_with_markdown.docx")
        )
        print(converter.summary())

    @staticmethod
    def main():
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
        llm = ChatOpenAI(model="gpt-4o-mini")
        prompt = ChatPromptTemplate.from_template(
            """
            당신은 한국 소득세 전문가입니다.

            아래 참고 문서에 제공된 정보를 근거로
            질문에 대해 단계적으로 생각하여 답변하세요.

            참고 문서에 명시된 자료를 활용해 결론을 도출하는 것은 허용됩니다.
            단, 참고 문서에 전혀 없는 새로운 사실을 만들어내지는 마세요.

            [참고 문서]
            {context}

            [질문]
            {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(
            database.as_retriever(search_kwargs={"k": 4}),
            document_chain,
        )
        query = "연봉 5천만원인 거주자의 소득세는 얼마인가요?"
        result = retrieval_chain.invoke({"input": query})
        print(f"{llm.__class__.__name__}:\n{result['answer']}\n")
