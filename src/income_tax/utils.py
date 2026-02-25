import numpy as np
import numpy.typing as npt
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec


class Utils:
    """
    # .env
    OPENAI_API_KEY="sk-proj-..."
    UPSTAGE_API_KEY="up_..."
    """

    @staticmethod
    def openai_vec(input: str, verbose=True) -> npt.NDArray:
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        embeddings = client.embeddings
        response = embeddings.create(
            input=input, model="text-embedding-3-large"  # suitable for Korean
        )
        vector = np.array(response.data[0].embedding)
        if verbose:
            print(f"{input}: {vector}")
        return vector

    @staticmethod
    def upstage_vec(input: str, verbose=True) -> npt.NDArray:
        load_dotenv()
        client = OpenAI(
            api_key=os.getenv("UPSTAGE_API_KEY"), base_url="https://api.upstage.ai/v1"
        )
        embeddings = client.embeddings
        response = embeddings.create(input=input, model="embedding-query")
        vector = np.array(response.data[0].embedding)
        if verbose:
            print(f"{input}: {vector}")
        return vector

    @staticmethod
    def cos_sim(vec1: npt.NDArray, vec2: npt.NDArray) -> float:
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

    @staticmethod
    def msg_content(llm: BaseChatModel, input: str):
        load_dotenv()
        ai_message = llm.invoke(input)
        print(f"{llm.__class__.__name__}:\n{ai_message.content}\n")

    @staticmethod
    def build_chroma_db(
        doc_path: Path,
        embedding: Embeddings,
        persist_dir: Path,
        collection_name="chroma-tax",
    ) -> Chroma:
        if persist_dir.exists():
            database = Chroma(
                collection_name=collection_name,
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
                collection_name=collection_name,
                persist_directory=str(persist_dir),
            )
        return database

    @staticmethod
    def build_pinecone_db(
        doc_path: Path, embedding: Embeddings, index_name="tax-index"
    ) -> PineconeVectorStore:
        pc = Pinecone()
        existing_indexes = [index["name"] for index in pc.list_indexes()]
        if index_name in existing_indexes:
            database = PineconeVectorStore(
                index_name=index_name,
                embedding=embedding,
            )
        else:
            loader = Docx2txtLoader(doc_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
            )
            document_list = loader.load_and_split(text_splitter=text_splitter)
            pc.create_index(
                name=index_name,
                dimension=len(embedding.embed_query("dimension check")),
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            database = PineconeVectorStore.from_documents(
                documents=document_list,
                embedding=embedding,
                index_name=index_name,
            )
        return database
