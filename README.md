# 📚 文献问答系统

基于RAG技术的PDF文献智能问答系统，使用智谱AI API回答文献相关问题。

## 系统功能

将PDF文献转换为向量数据库，通过语义检索找到相关内容，结合大模型生成准确答案并显示引用来源。

## 使用流程

### 1. 配置环境

```bash
# 激活虚拟环境
source venv/Scripts/activate

# 配置API密钥（编辑.env文件）
ZHIPU_API_KEY=你的密钥
```

### 2. 处理文献

将PDF文件放入 `documents/` 文件夹，然后运行：

```bash
python process_documents.py
```

### 3. 启动问答

```bash
streamlit run app.py
```

访问 http://localhost:8501 开始提问。

## 核心文件说明

### 主程序

- **app.py** - Streamlit问答界面
  - `load_vector_store()`: 加载向量数据库
  - `get_zhipu_response()`: 调用智谱AI生成回答
  - `main()`: 界面主函数

- **process_documents.py** - 文档处理脚本
  - `create_vector_store()`: 创建向量数据库
  - `main()`: 批量处理PDF文献

### 工具模块

- **pdf_processor.py** - PDF处理类
  - `load_pdf()`: 读取单个PDF
  - `load_documents()`: 批量加载并分块

- **vector_store.py** - 向量数据库类（可选）
  - `create_vectorstore()`: 创建数据库
  - `similarity_search()`: 相似度检索

- **qa_chain.py** - 问答链类（可选）
  - `ask()`: 问答接口

## 技术架构

```
PDF文献 → 文本提取 → 分块 → 向量化 → ChromaDB
                                        ↓
用户提问 → 语义检索 → 相关文本 → 智谱AI → 生成回答
```

## 依赖说明

- LangChain: 文档处理框架
- ChromaDB: 向量数据库
- Sentence Transformers: 文本嵌入
- 智谱AI: 大语言模型
- Streamlit: Web界面