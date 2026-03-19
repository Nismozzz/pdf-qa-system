"""
问答链模块
负责构建问答链并生成回答
"""
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate

# 加载环境变量
load_dotenv()


class QAChain:
    def __init__(self, vectorstore, model_type: str = "openai"):
        """
        初始化问答链

        Args:
            vectorstore: 向量数据库实例
            model_type: 模型类型 (openai, dashscope等)
        """
        self.vectorstore = vectorstore
        self.model_type = model_type
        self.llm = self._init_llm()
        self.qa_chain = self._create_qa_chain()

    def _init_llm(self):
        """
        初始化大语言模型
        """
        if self.model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("请在.env文件中设置OPENAI_API_KEY")
            return OpenAI(temperature=0, openai_api_key=api_key)
        # 可以在这里添加其他模型的支持
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _create_qa_chain(self):
        """
        创建问答链
        """
        # 自定义提示模板
        template = """你是一个专业的学术助手。请基于以下文献内容回答问题。
        如果文献中没有相关信息，请明确说明"根据提供的文献，我无法回答这个问题"。

        文献内容：
        {context}

        问题：{question}

        请提供详细且准确的回答："""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        return qa_chain

    def ask(self, question: str) -> dict:
        """
        提问并获取回答

        Args:
            question: 用户问题

        Returns:
            包含答案和来源文档的字典
        """
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }
