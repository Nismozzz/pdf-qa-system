import os
import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def load_vector_store():
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': embedding_device},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
    vectorstore = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings
    )
    return vectorstore

def get_zhipu_response(question, context):
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    model_name = os.getenv("LLM_MODEL", "glm-4-flash")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    prompt = f"""基于以下文献内容回答问题。如果文献中没有相关信息，请说明。

文献内容：
{context}

问题：{question}

请用中文回答："""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content

vectorstore = None

def answer_question(question):
    global vectorstore
    if not question.strip():
        yield "请输入问题", ""
        return

    if vectorstore is None:
        yield "向量数据库未找到，请先运行: python process_documents.py", ""
        return

    yield "正在检索相关文献...", ""

    docs = vectorstore.similarity_search(question, k=3)
    sources = ""
    for i, doc in enumerate(docs, 1):
        sources += f"**来源 {i}：{doc.metadata['source']}**\n"
        sources += doc.page_content[:300] + "...\n\n---\n\n"

    yield "正在生成回答，请稍候...", sources

    context = "\n\n".join([f"来源：{doc.metadata['source']}\n{doc.page_content}"
                          for doc in docs])
    answer = get_zhipu_response(question, context)

    yield answer, sources

def main():
    global vectorstore

    vector_db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
    if os.path.exists(vector_db_path):
        vectorstore = load_vector_store()

    with gr.Blocks(title="文献问答系统") as demo:
        gr.Markdown("# 文献问答系统")

        question = gr.Textbox(
            label="请输入你的问题",
            placeholder="例如：这些文献的主要研究方法是什么？",
        )
        submit_btn = gr.Button("提问", variant="primary")

        answer_output = gr.Markdown(label="回答")

        with gr.Accordion("查看引用来源", open=False):
            sources_output = gr.Markdown(label="引用来源")

        submit_btn.click(
            fn=answer_question,
            inputs=question,
            outputs=[answer_output, sources_output],
        )
        question.submit(
            fn=answer_question,
            inputs=question,
            outputs=[answer_output, sources_output],
        )

    demo.launch()

if __name__ == "__main__":
    main()
