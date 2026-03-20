# 文献问答系统

基于 RAG（检索增强生成）技术的 PDF 文献智能问答系统。将 PDF 文献向量化存储，通过语义检索找到相关内容，结合大语言模型生成回答并展示引用来源。

## 技术架构

```
PDF文献 --> 文本提取 --> 分块 --> 向量化 --> ChromaDB
                                              |
用户提问 --> 语义检索 --> 相关文本 --> LLM --> 生成回答
```

- LangChain：文档处理与向量化框架
- ChromaDB：向量数据库
- Sentence Transformers：文本嵌入模型（默认paraphrase-multilingual-MiniLM-L12-v2）
- Gradio：Web 交互界面
- 兼容 OpenAI 接口格式的大语言模型（默认智谱 GLM）

## 使用流程

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填入你的配置。

### 3. 处理文献

将 PDF 文件放入 `documents/` 文件夹（需要新建），然后运行：

```bash
python process_documents.py
```

该脚本会读取所有 PDF、分块、向量化，并将结果存入 `vector_db/` 目录。

### 4. 启动问答系统

```bash
python app.py
```

启动后访问终端输出的本地地址（默认 http://localhost:7860），即可在网页上输入问题进行问答。

## 核心文件说明

### app.py -- 主程序（Gradio 界面）

- `load_vector_store()`：加载已有的向量数据库和嵌入模型，返回 ChromaDB 实例。
- `get_zhipu_response(question, context)`：将用户问题和检索到的文献内容拼接为 prompt，调用 LLM API 生成回答。
- `answer_question(question)`：完整的问答流程。先检索相关文档，再调用 LLM 生成回答，以生成器方式逐步返回状态和结果。
- `main()`：构建 Gradio 界面并启动 Web 服务。

### process_documents.py -- 文档预处理脚本

- `create_vector_store(documents)`：接收文档列表，加载嵌入模型，创建 ChromaDB 向量数据库并持久化。
- `main()`：调用 PDFProcessor 批量加载 `documents/` 目录下的 PDF 文件，完成分块和向量化。

### pdf_processor.py -- PDF 处理模块

- `PDFProcessor.__init__(chunk_size, chunk_overlap)`：初始化文本分块器，`chunk_size` 控制每块字符数，`chunk_overlap` 控制块间重叠字符数。
- `load_pdf(pdf_path)`：读取单个 PDF 文件，提取全部页面的文本内容。
- `load_documents(documents_dir)`：遍历目录下所有 PDF 文件，逐一提取文本并分块，返回 LangChain Document 列表。

### vector_store.py -- 向量数据库模块

- `VectorStore.__init__(persist_directory)`：初始化嵌入模型和存储路径。
- `create_vectorstore(documents)`：从文档列表创建 ChromaDB 向量数据库。
- `load_vectorstore()`：加载已存在的向量数据库。
- `similarity_search(query, k)`：在向量数据库中进行相似度检索，返回最相关的 k 个文档。

### qa_chain.py -- 问答链模块（备用）

- `QAChain.__init__(vectorstore, model_type)`：初始化 LLM 和 LangChain 问答链。
- `ask(question)`：输入问题，返回包含回答和来源文档的字典。

> 注：当前主程序 `app.py` 未使用 `qa_chain.py` 和 `vector_store.py` 中的类，而是直接实现了检索和问答逻辑。这两个模块作为可复用的组件保留。

## 注意事项

1. **文献格式**：仅支持 PDF 格式。扫描件类型的 PDF（图片型）可能无法正确提取文本。
2. **向量数据库**：添加新文献后需重新运行 `process_documents.py`，会覆盖已有的向量数据库。
