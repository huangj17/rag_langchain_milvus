# Milvus RAG 工具集

专为 RAG 系统设计的 Milvus 向量数据库工具类集合。

## 文件说明

### 1. `milvus_util.py` - 核心工具类

封装了 Milvus 向量数据库的所有操作，提供简洁的 API 接口。

**主要功能：**

- 🔌 数据库连接管理
- 🧠 Embedding 模型初始化（支持 Ollama）
- 📦 向量存储创建与配置
- 📝 文档添加（支持批量、元数据）
- 🔍 相似度搜索
- 🎯 MMR 搜索（多样性优化）
- 🛠️ 集合管理（创建、删除、查询信息）

**快速开始：**

```python
from milvus_util import MilvusUtil

# 初始化并连接
milvus = MilvusUtil(
    db_name="my_rag_db",
    collection_name="documents",
    verbose=True
)

# 一键初始化（连接、数据库、embedding、向量存储）
milvus.initialize()

# 添加文档
texts = ["文档1", "文档2", "文档3"]
milvus.add_texts(texts)

# 搜索
results = milvus.similarity_search("查询内容", k=3)

# 关闭连接
milvus.close()
```

### 2. `index.py` - RAG 索引构建模块

提供完整的文档加载、分割、向量化和索引构建流程。

**支持的文档格式：**

- 📄 PDF (.pdf)
- 📝 Markdown (.md)
- 📃 Text (.txt, .log)
- 📊 CSV (.csv)
- 🗂️ JSON (.json)
- 🌐 HTML (.html, .htm)
- 📑 Word (.docx, .doc)

**使用示例：**

```python
from index import build_rag_index

# 构建完整的RAG索引
milvus = build_rag_index(
    data_dir="./data",              # 文档目录
    db_name="rag_database",         # 数据库名
    collection_name="my_docs",      # 集合名
    chunk_size=500,                 # 分块大小
    chunk_overlap=50,               # 重叠大小
    drop_old=True,                  # 是否重建
    verbose=True
)

# 使用构建好的索引进行搜索
if milvus:
    results = milvus.similarity_search("查询问题", k=3)
    milvus.close()
```

### 3. `example_usage.py` - 使用示例

包含 5 个详细的使用示例：

1. **基本使用** - 连接、添加、搜索的完整流程
2. **MMR 搜索** - 提升结果多样性的搜索方式
3. **LangChain 集成** - 与文档加载器结合使用
4. **集合管理** - 查看和管理 Milvus 集合
5. **自定义参数** - 高级索引和搜索参数配置

## 核心类：MilvusUtil

### 初始化参数

| 参数                 | 类型 | 默认值                   | 说明                 |
| -------------------- | ---- | ------------------------ | -------------------- |
| `host`               | str  | "127.0.0.1"              | Milvus 服务器地址    |
| `port`               | int  | 19530                    | Milvus 服务器端口    |
| `db_name`            | str  | "milvus_demo"            | 数据库名称           |
| `embedding_model`    | str  | "qwen3-embedding:4b"     | Embedding 模型       |
| `embedding_base_url` | str  | "http://localhost:11434" | Embedding 服务地址   |
| `collection_name`    | str  | None                     | 集合名称（可选）     |
| `drop_old`           | bool | False                    | 是否删除已存在的集合 |
| `verbose`            | bool | False                    | 是否显示详细日志     |

### 主要方法

#### 连接与初始化

```python
# 单步初始化
milvus.connect()              # 连接数据库
milvus.setup_database()       # 设置数据库
milvus.init_embeddings()      # 初始化embedding
milvus.create_vector_store()  # 创建向量存储

# 或一键初始化
milvus.initialize()           # 执行上述所有步骤
```

#### 文档操作

```python
# 方式1: 添加Document对象（推荐用于RAG场景）
# 适用于已通过LangChain加载器处理的文档
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("document.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
splits = splitter.split_documents(documents)

# 直接添加Document对象，保留所有元数据
milvus.add_documents(splits)

# 方式2: 添加纯文本（适用于简单场景）
milvus.add_texts(
    texts=["文本1", "文本2"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}]
)
```

**方法对比：**

| 方法              | 适用场景             | 优点                                 | 输入格式                   |
| ----------------- | -------------------- | ------------------------------------ | -------------------------- |
| `add_documents()` | RAG 系统、多文件处理 | 自动保留文档来源和元数据，代码更简洁 | `List[Document]`           |
| `add_texts()`     | 简单文本添加         | 灵活性高，可以只添加文本             | `List[str]` + `List[Dict]` |

#### 搜索

```python
# 相似度搜索
results = milvus.similarity_search(
    query="查询内容",
    k=3,                    # 返回top-k结果
    filter={"source": "doc1"}  # 可选的元数据过滤
)

# MMR搜索（更多样化的结果）
results = milvus.mmr_search(
    query="查询内容",
    k=3,                    # 最终返回数量
    fetch_k=20,             # 候选数量
    lambda_mult=0.5         # 多样性参数(0-1)
)
```

#### 集合管理

```python
# 获取集合信息
info = milvus.get_collection_info()
print(info)  # {"name": "...", "vector_dim": 768, ...}

# 删除集合
milvus.drop_collection("collection_name")
```

#### 关闭连接

```python
milvus.close()
```

## 配置说明

### 索引参数（HNSW）

```python
index_params = {
    "index_type": "HNSW",        # 索引类型
    "metric_type": "L2",         # 距离度量（L2或IP）
    "params": {
        "M": 16,                 # 连接数(10-32)，越大精度越高
        "efConstruction": 200,   # 构建参数(100-300)
    }
}
```

### 搜索参数

```python
search_params = {
    "metric_type": "L2",
    "params": {
        "ef": 64                 # 搜索范围(40-128)，越大精度越高
    }
}
```

## 完整工作流示例

```python
from milvus_util import MilvusUtil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载文档
loader = TextLoader("document.txt")
documents = loader.load()

# 2. 分割文档
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = splitter.split_documents(documents)

# 3. 初始化Milvus
milvus = MilvusUtil(
    db_name="my_knowledge_base",
    collection_name="documents",
    drop_old=True,  # 重建索引
    verbose=True
)

if not milvus.initialize():
    print("初始化失败")
    exit(1)

# 4. 添加文档
texts = [doc.page_content for doc in splits]
metadatas = [doc.metadata for doc in splits]
milvus.add_texts(texts, metadatas)

# 5. 进行查询
query = "这个文档讲了什么？"
results = milvus.mmr_search(query, k=3, fetch_k=20)

for i, doc in enumerate(results, 1):
    print(f"\n结果 {i}:")
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")

# 6. 清理
milvus.close()
```

## 依赖要求

```bash
pip install langchain-milvus
pip install langchain-ollama
pip install pymilvus
pip install langchain-community
pip install langchain-text-splitters
```

## 注意事项

1. **Milvus 服务** - 确保 Milvus 服务已启动（默认端口 19530）
2. **Ollama 服务** - 确保 Ollama 服务已启动（默认端口 11434）
3. **Embedding 模型** - 确保已下载所需的 embedding 模型（如 qwen3-embedding:4b）
4. **集合管理** - 设置`drop_old=True`会删除已有集合，请谨慎使用
5. **日志输出** - 设置`verbose=True`可查看详细执行过程

## 性能优化建议

### 索引参数调优

- **高精度场景**：增大 M (如 32) 和 efConstruction (如 300)
- **高性能场景**：减小 M (如 8) 和 efConstruction (如 100)
- **平衡场景**：M=16, efConstruction=200（默认）

### 搜索参数调优

- **高召回率**：增大 ef (如 128)
- **快速搜索**：减小 ef (如 32)
- **平衡**：ef=64（默认）

### 文档分块建议

- **短文档**：chunk_size=200-300
- **中等文档**：chunk_size=500-800
- **长文档**：chunk_size=1000-1500
- **重叠比例**：一般为 chunk_size 的 10%-20%

## 故障排除

### 连接失败

```bash
# 检查Milvus服务状态
docker ps | grep milvus

# 检查端口是否开放
telnet 127.0.0.1 19530
```

### Embedding 失败

```bash
# 检查Ollama服务
curl http://localhost:11434/api/tags

# 拉取模型
ollama pull qwen3-embedding:4b
```

### 搜索无结果

1. 确认文档已成功添加：`milvus.get_collection_info()`
2. 检查集合是否为空：`info['is_empty']`
3. 尝试使用不同的查询文本

## License

MIT
