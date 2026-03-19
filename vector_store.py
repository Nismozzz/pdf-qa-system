"""
向量数据库模块
负责文档向量化和相似度检索
"""
import os
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()


class VectorStore:
    def __init__(self, persist_directory: str = "./vector_db"):
        """
        初始化向量数据库

        Args:
            persist_directory: 向量数据库持久化目录
        """
        self.persist_directory = persist_directory
        # 使用本地的中文embedding模型
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': embedding_device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None

    def create_vectorstore(self, documents: List[Document]):
        """
        创建向量数据库

        Args:
            documents: 文档列表
        """
        print("正在创建向量数据库...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("向量数据库创建完成！")

    def load_vectorstore(self):
        """
        加载已存在的向量数据库
        """
        if os.path.exists(self.persist_directory):
            print("正在加载向量数据库...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("向量数据库加载完成！")
            return True
        return False

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回最相似的k个文档

        Returns:
            相关文档列表
        """
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化")

        return self.vectorstore.similarity_search(query, k=k)
