# MilvusUtil 使用指南

## 快速开始

### 基础使用

```python
from milvus_util import MilvusUtil

# 1. 创建实例（使用默认配置）
util = MilvusUtil(verbose=True)

# 2. 一键初始化
if util.initialize():
    print("初始化成功！")
else:
    print("初始化失败")

# 3. 添加文档
from langchain.schema import Document

documents = [
    Document(page_content="这是第一个文档", metadata={"source": "doc1.txt"}),
    Document(page_content="这是第二个文档", metadata={"source": "doc2.txt"}),
]

util.add_documents(documents)

# 4. 创建RAG工作流并查询
graph = util.similarity_search(query="测试查询", k=3)

if graph:
    # 执行查询
    result = graph.invoke({"messages": [{"role": "user", "content": "你的问题"}]})
    print(result)

# 5. 关闭连接
util.close()
```

---

## 自定义配置

### 方式 1：初始化时配置

```python
util = MilvusUtil(
    # 数据库配置
    host="192.168.1.100",
    port=19530,
    db_name="my_database",

    # Embedding配置
    embedding_model="qwen3-embedding:4b",
    embedding_base_url="http://localhost:11434",

    # LLM配置
    llm_model="deepseek-v3.1:671b",
    llm_base_url="https://ollama.com",

    # 集合配置
    collection_name="my_collection",
    drop_old=False,  # 是否删除旧集合

    # 日志配置
    verbose=True
)
```

### 方式 2：修改默认常量

```python
from milvus_util import MilvusUtil
import milvus_util

# 修改默认配置
milvus_util.DEFAULT_LLM_MODEL = "llama3"
milvus_util.DEFAULT_HNSW_M = 32
milvus_util.DEFAULT_SEARCH_K = 5

# 创建实例（使用修改后的默认值）
util = MilvusUtil()
```

---

## 分步初始化

如果需要更精细的控制，可以分步初始化：

```python
util = MilvusUtil(verbose=True)

# 步骤1：连接数据库
if not util.connect():
    print("连接失败")
    exit(1)

# 步骤2：设置数据库
if not util.setup_database():
    print("数据库设置失败")
    exit(1)

# 步骤3：初始化Embedding模型
if not util.init_embeddings():
    print("Embedding初始化失败")
    exit(1)

# 步骤4：创建向量存储
# 自定义索引参数
custom_index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",  # 使用余弦相似度
    "params": {
        "M": 32,
        "efConstruction": 400,
    },
}

if not util.create_vector_store(index_params=custom_index_params):
    print("向量存储创建失败")
    exit(1)

# 步骤5：初始化LLM
if not util.init_response_model(
    model="custom-model",
    temperature=0.5
):
    print("LLM初始化失败")
    exit(1)

print("所有组件初始化完成！")
```

---

## 文档管理

### 添加文本

```python
# 直接添加文本
texts = ["文本1", "文本2", "文本3"]
metadatas = [
    {"source": "file1.txt", "page": 1},
    {"source": "file2.txt", "page": 1},
    {"source": "file3.txt", "page": 1},
]

util.add_texts(texts, metadatas=metadatas)
```

### 添加文档对象

```python
from langchain.schema import Document

documents = [
    Document(
        page_content="文档内容1",
        metadata={"source": "doc1.txt", "author": "张三"}
    ),
    Document(
        page_content="文档内容2",
        metadata={"source": "doc2.txt", "author": "李四"}
    ),
]

util.add_documents(documents)
```

---

## 检索方法

### 1. 相似度搜索（含 RAG 工作流）

```python
# 创建RAG工作流图
graph = util.similarity_search(query="测试", k=5)

# 执行查询
if graph:
    result = graph.invoke({
        "messages": [{"role": "user", "content": "你的问题"}]
    })

    # 获取最终答案
    final_message = result["messages"][-1]
    print(final_message.content)
```

### 2. MMR 搜索（多样性检索）

```python
# MMR搜索，减少冗余
results = util.mmr_search(
    query="查询文本",
    k=5,              # 返回5个结果
    fetch_k=20,       # 先获取20个候选
    lambda_mult=0.3   # 多样性参数（0-1，越小越多样）
)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("---")
```

---

## 集合管理

### 获取集合信息

```python
info = util.get_collection_info()

if info:
    print(f"集合名称: {info['name']}")
    print(f"是否为空: {info['is_empty']}")
    print(f"向量维度: {info['vector_dim']}")
    print(f"索引信息: {info['indexes']}")
```

### 删除集合

```python
# 删除当前集合
util.drop_collection()

# 删除指定集合
util.drop_collection("my_collection")
```

---

## RAG 工作流自定义

### 自定义 Prompt 模板

```python
import milvus_util

# 在创建实例前修改Prompt模板
milvus_util.GENERATE_PROMPT_TEMPLATE = """
你是一个专业的AI助手。请根据以下上下文回答问题。

上下文信息：
{context}

用户问题：
{question}

请提供详细且准确的回答：
"""

# 创建实例并使用自定义Prompt
util = MilvusUtil()
util.initialize()
```

### 工作流图可视化

```python
# 创建工作流
graph = util.similarity_search(query="测试", k=3)

# 工作流图会自动保存为 workflow_graph.png
# 可以打开查看工作流结构
```

---

## 最佳实践

### 1. 使用上下文管理器

虽然当前版本不支持上下文管理器，但可以这样使用：

```python
def use_milvus():
    util = MilvusUtil(verbose=True)
    try:
        if not util.initialize():
            raise Exception("初始化失败")

        # 使用工具类
        util.add_texts(["文本1", "文本2"])
        graph = util.similarity_search(query="测试", k=3)

        return graph
    finally:
        util.close()  # 确保连接被关闭

graph = use_milvus()
```

### 2. 批量添加文档

```python
from langchain.document_loaders import DirectoryLoader, TextLoader

# 加载目录中的所有文档
loader = DirectoryLoader(
    "./data",
    glob="**/*.txt",
    loader_cls=TextLoader
)

documents = loader.load()

# 分批添加（避免一次性加载过多）
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    util.add_documents(batch)
    print(f"已添加 {i+batch_size}/{len(documents)} 个文档")
```

### 3. 错误处理

```python
util = MilvusUtil(verbose=True)

try:
    if not util.initialize():
        raise Exception("初始化失败")

    # 添加文档
    success = util.add_documents(documents)
    if not success:
        print("文档添加失败")

    # 搜索
    graph = util.similarity_search(query="测试", k=5)
    if graph is None:
        print("搜索失败")
    else:
        result = graph.invoke({"messages": [{"role": "user", "content": "问题"}]})
        print(result)

except Exception as e:
    print(f"发生错误: {e}")
finally:
    util.close()
```

---

## 高级配置

### 自定义索引参数

```python
# IVF_FLAT 索引
ivf_index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {
        "nlist": 128,  # 聚类中心数量
    },
}

ivf_search_params = {
    "metric_type": "L2",
    "params": {
        "nprobe": 10,  # 搜索的聚类数量
    },
}

util = MilvusUtil()
util.connect()
util.setup_database()
util.init_embeddings()
util.create_vector_store(
    index_params=ivf_index_params,
    search_params=ivf_search_params
)
```

### 环境变量配置

```python
import os

# 设置API密钥
os.environ["OLLAMA_API_KEY"] = "your-api-key"

# 创建实例
util = MilvusUtil()
```

---

## 常见问题

### Q: 如何修改默认的搜索结果数量？

```python
# 方式1：在搜索时指定
graph = util.similarity_search(query="测试", k=10)

# 方式2：修改默认值
import milvus_util
milvus_util.DEFAULT_SEARCH_K = 10
```

### Q: 如何使用不同的距离度量？

```python
# 创建向量存储时指定
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",  # 使用余弦相似度
    "params": {"M": 16, "efConstruction": 200},
}

search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 64},
}

util.create_vector_store(
    index_params=index_params,
    search_params=search_params
)
```

### Q: 如何禁用 verbose 日志？

```python
util = MilvusUtil(verbose=False)
```

### Q: 如何复用已有的集合？

```python
# 不删除旧集合
util = MilvusUtil(
    collection_name="existing_collection",
    drop_old=False  # 保留旧数据
)
```

---

## 性能优化建议

1. **批量操作**：一次添加多个文档比逐个添加更高效
2. **合适的索引参数**：根据数据量调整 `M` 和 `efConstruction`
3. **搜索参数调优**：平衡 `ef` 值以获得速度和准确率的最佳组合
4. **使用 MMR**：对于需要多样性的场景，使用 MMR 搜索
5. **连接复用**：避免频繁创建和销毁连接

---

## 更多资源

- [Milvus 官方文档](https://milvus.io/docs)
- [LangChain 文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)

---

## 技术支持

如有问题，请查看：

1. `REFACTORING_SUMMARY.md` - 了解代码优化细节
2. `README.md` - 项目总体说明
3. 代码中的文档字符串 - 查看详细的方法说明
