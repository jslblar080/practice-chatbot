import numpy as np
import os

from dotenv import load_dotenv
from openai import OpenAI


class Utils:
    """
    # .env
    OPENAI_API_KEY="sk-proj-..."
    UPSTAGE_API_KEY="up_..."
    """

    @staticmethod
    def openai_vec(input: str, verbose=True) -> np.typing.NDArray:
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=input, model="text-embedding-3-large"  # suitable for Korean
        )
        vector = np.array(response.data[0].embedding)
        if verbose:
            print(f"{input}: {vector}")
        return vector

    @staticmethod
    def upstage_vec(input: str, verbose=True) -> np.typing.NDArray:
        load_dotenv()
        client = OpenAI(
            api_key=os.getenv("UPSTAGE_API_KEY"), base_url="https://api.upstage.ai/v1"
        )
        response = client.embeddings.create(input=input, model="embedding-query")
        vector = np.array(response.data[0].embedding)
        if verbose:
            print(f"{input}: {vector}")
        return vector

    @staticmethod
    def cos_sim(vec1: np.typing.NDArray, vec2: np.typing.NDArray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            print("0.0")
            return 0.0
        result = dot_product / (norm_vec1 * norm_vec2)
        print(f"cosine similarity: {result}")
        return result

    @classmethod
    def show_sim(cls, word1: str, word2: str, client_type="upstage") -> float:
        if client_type == "upstage":
            emb_vec = cls.upstage_vec
        elif client_type == "openai":
            emb_vec = cls.openai_vec
        else:
            client_type = "upstage"
            emb_vec = cls.upstage_vec
        print(f"Client using {client_type}")
        word1_vec = emb_vec(word1)
        word2_vec = emb_vec(word2)
        word1_word2_similarity = cls.cos_sim(word1_vec, word2_vec)
        print()
        return word1_word2_similarity
