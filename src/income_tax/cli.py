from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage
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
        return
