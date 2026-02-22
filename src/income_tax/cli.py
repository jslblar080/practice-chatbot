from .utils import Utils


class CLI:

    @staticmethod
    def main():
        Utils.show_sim("king", "왕", client_type="openai")
        Utils.show_sim("king", "왕", client_type="upstage")
