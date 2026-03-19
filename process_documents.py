import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from pdf_processor import PDFProcessor

load_dotenv()

def create_vector_store(documents):
    """创建向量数据库"""
    print("正在创建向量数据库...")
    print("正在加载嵌入模型（首次运行会下载模型，请耐心等待）...")

    embeddings = HuggingFaceEmbeddings(
        model_name=r"D:\大学\大四上\学校实习\车站项目\车站项目_新\models\paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=vector_db_path
    )
    print(f"向量数据库已创建: {vector_db_path}")
    return vectorstore

def main():
    print("=" * 50)
    print("PDF文档处理程序")
    print("=" * 50)

    try:
        # 使用PDFProcessor类加载和分割文档
        processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
        documents = processor.load_documents("documents")

        if not documents:
            print("错误: 请将PDF文件放入 documents 文件夹")
            return

        # 创建向量数据库
        create_vector_store(documents)
        print("\n处理完成！运行: python app.py")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确保已安装所有依赖: pip install -r requirements.txt")
        print("3. 如果是网络问题，可以尝试使用代理或稍后重试")

if __name__ == "__main__":
    main()
