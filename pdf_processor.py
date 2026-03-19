"""
PDF文档处理模块
负责读取PDF文件并提取文本内容
"""
import os
from typing import List
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化PDF处理器

        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_pdf(self, pdf_path: str) -> str:
        """
        读取单个PDF文件

        Args:
            pdf_path: PDF文件路径

        Returns:
            提取的文本内容
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            print(f"读取PDF失败 {pdf_path}: {str(e)}")
            return ""

    def load_documents(self, documents_dir: str) -> List[Document]:
        """
        批量加载目录下的所有PDF文件

        Args:
            documents_dir: 文档目录路径

        Returns:
            文档列表
        """
        documents = []

        if not os.path.exists(documents_dir):
            print(f"目录不存在: {documents_dir}")
            return documents

        for filename in os.listdir(documents_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(documents_dir, filename)
                print(f"正在处理: {filename}")

                text = self.load_pdf(pdf_path)
                if text:
                    # 将文本分块
                    chunks = self.text_splitter.split_text(text)
                    # 创建Document对象
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": filename,
                                "chunk": i
                            }
                        )
                        documents.append(doc)

        print(f"共加载 {len(documents)} 个文本块")
        return documents
