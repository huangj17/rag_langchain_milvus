"""
Milvus数据库工具类

功能：
1. 连接和管理Milvus数据库
2. 初始化Embedding模型
3. 创建和管理向量存储
4. 文档添加和相似性搜索
5. 集合管理
6. RAG工作流集成

设计理念：
- 封装为工具类，便于RAG系统集成
- 支持灵活配置（数据库、集合、索引参数等）
- 自动管理连接和资源
- 提供清晰的API接口
- 职责分离，模块化设计

备注：
1. 文本嵌入模型使用本地ollama服务
2. 大语言模型使用ollama云服务(免费api)
"""

import logging
import os
from typing import Any, Dict, List, Literal, Optional

from IPython.display import Image, display
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pymilvus import Collection, MilvusException, connections, db, utility

# ==================== 常量定义 ====================

# 默认配置
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 19530
DEFAULT_DB_NAME = "milvus_demo"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:4b"
DEFAULT_EMBEDDING_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "deepseek-v3.1:671b"
DEFAULT_LLM_BASE_URL = "https://ollama.com"
DEFAULT_TEMPERATURE = 0.7

# 索引配置
DEFAULT_INDEX_TYPE = "HNSW"
DEFAULT_METRIC_TYPE = "L2"
DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_EF = 64

# 搜索配置
DEFAULT_SEARCH_K = 3
DEFAULT_MMR_FETCH_K = 20
DEFAULT_MMR_LAMBDA = 0.5

# Prompt 模板
GRADE_PROMPT_TEMPLATE = (
    "你是一名评审员，需要判断检索到的文档与用户问题的相关性。\n\n"
    "检索到的文档：\n{context}\n\n"
    "用户问题：{question}\n\n"
    "请判断文档是否与问题相关。如果文档包含与问题相关的关键词或语义信息，判定为相关。\n"
    "只需回答 'yes' 或 'no'，不要有其他内容。"
)

REWRITE_PROMPT_TEMPLATE = (
    "请审视输入内容，并尽量推理其潜在的语义意图。\n"
    "这是最初的问题："
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "请将其改写为更优的问题："
)

GENERATE_PROMPT_TEMPLATE = (
    "你是一名问答助手。请利用以下检索到的上下文片段来回答问题。"
    "如果你不知道答案，就直接说你不知道。"
    "答案回复尽量详细，不要过于简洁。\n"
    "问题: {question} \n"
    "上下文: {context}"
)

# RAG 工作流相关配置
## 工具名称：用于检索相关文档
RETRIEVER_TOOL_NAME = "retrieve_documents"
## 工具描述：用于工作流工具集成文档搜索功能
RETRIEVER_TOOL_DESCRIPTION = "搜索并返回相关文档"
## 默认工作流图保存路径
DEFAULT_GRAPH_PATH = "./workflow_graph.png"

# 环境变量 ollama api key
if not os.environ.get("OLLAMA_API_KEY"):
    os.environ["OLLAMA_API_KEY"] = "xxxxx"

# 配置日志
logger = logging.getLogger(__name__)


class MilvusUtil:
    """
    Milvus数据库工具类，用于RAG系统的向量存储

    主要功能：
    - 数据库连接管理
    - 向量存储和检索
    - RAG工作流集成
    - 文档管理和搜索
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        db_name: str = DEFAULT_DB_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
        llm_model: str = DEFAULT_LLM_MODEL,
        llm_base_url: str = DEFAULT_LLM_BASE_URL,
        collection_name: Optional[str] = None,
        drop_old: bool = False,
        verbose: bool = False,
    ):
        """
        初始化Milvus工具类

        Args:
            host: Milvus服务器地址
            port: Milvus服务器端口
            db_name: 数据库名称
            embedding_model: Embedding模型名称
            embedding_base_url: Embedding服务地址
            llm_model: LLM模型名称
            llm_base_url: LLM服务地址
            collection_name: 集合名称（可选，不指定则使用默认）
            drop_old: 是否删除已存在的集合
            verbose: 是否输出详细日志
        """
        # 数据库配置
        self.host = host
        self.port = port
        self.db_name = db_name

        # Embedding配置
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url

        # LLM配置
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url

        # 集合配置
        self.collection_name = collection_name
        self.drop_old = drop_old

        # 运行时配置
        self.verbose = verbose

        # 内部状态
        self._connected = False
        self.embeddings: Optional[OllamaEmbeddings] = None
        self.vector_store: Optional[Milvus] = None
        self.llm: Optional[ChatOllama] = None
        self.retriever_tool = None

        # 配置日志级别
        if verbose:
            logging.basicConfig(level=logging.INFO)

    # ==================== 连接管理 ====================

    def connect(self) -> bool:
        """
        连接到Milvus数据库

        Returns:
            bool: 连接是否成功
        """
        try:
            connections.connect(host=self.host, port=self.port)

            if connections.has_connection("default"):
                self._connected = True
                if self.verbose:
                    logger.info(f"✅ 成功连接到Milvus: {self.host}:{self.port}")
                return True
            else:
                logger.error("❌ 连接失败: 未建立默认连接")
                return False
        except Exception as e:
            logger.error(f"❌ Milvus连接失败: {e}")
            return False

    def close(self) -> None:
        """关闭数据库连接"""
        try:
            if self._connected:
                connections.disconnect("default")
                self._connected = False
                if self.verbose:
                    logger.info("✅ 已断开Milvus连接")
        except Exception as e:
            logger.error(f"❌ 断开连接失败: {e}")

    # ==================== 数据库管理 ====================

    def setup_database(self) -> bool:
        """
        设置数据库（存在则复用，不存在则创建）

        Returns:
            bool: 操作是否成功
        """
        try:
            existing_databases = db.list_database()

            if self.db_name in existing_databases:
                if self.verbose:
                    logger.info(f"数据库 '{self.db_name}' 已存在，直接使用")
            else:
                db.create_database(self.db_name)
                if self.verbose:
                    logger.info(f"✅ 创建数据库 '{self.db_name}'")

            # 切换到指定数据库
            db.using_database(self.db_name)

            if self.verbose:
                collections = utility.list_collections()
                if collections:
                    logger.info(f"当前数据库中的集合: {collections}")

            return True
        except MilvusException as e:
            logger.error(f"❌ 数据库操作失败: {e}")
            return False

    # ==================== 模型初始化 ====================

    def init_embeddings(self) -> bool:
        """
        初始化Embedding模型

        Returns:
            bool: 初始化是否成功
        """
        try:
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.embedding_base_url,
            )

            if self.verbose:
                # 测试embedding
                test_embedding = self.embeddings.embed_query("测试")
                logger.info(
                    f"✅ Embedding模型初始化成功，向量维度: {len(test_embedding)}"
                )

            return True
        except Exception as e:
            logger.error(f"❌ Embedding模型初始化失败: {e}")
            return False

    def init_response_model(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> bool:
        """
        初始化响应模型（LLM）

        Args:
            model: 模型名称（可选，默认使用初始化时的配置）
            base_url: 服务地址（可选，默认使用初始化时的配置）
            temperature: 温度参数

        Returns:
            bool: 初始化是否成功
        """
        try:
            model_name = model or self.llm_model
            service_url = base_url or self.llm_base_url

            self.llm = ChatOllama(
                model=model_name,
                base_url=service_url,
                temperature=temperature,
                headers={"Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"},
            )

            if self.verbose:
                logger.info(f"✅ LLM模型初始化成功: {model_name}")

            return True
        except Exception as e:
            logger.error(f"❌ 初始化响应模型失败: {e}")
            return False

    # ==================== 向量存储管理 ====================

    def create_vector_store(
        self,
        index_params: Optional[Dict[str, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        创建向量存储

        Args:
            index_params: 索引参数（可选）
                - index_type: 索引类型 ("HNSW", "IVF_FLAT", "AUTOINDEX")
                - metric_type: 距离度量 ("L2", "IP", "COSINE")
                - params: 索引特定参数
            search_params: 搜索参数（可选）
                - metric_type: 距离度量方式
                - params: 搜索特定参数

        Returns:
            bool: 创建是否成功
        """
        if not self.embeddings:
            logger.error("❌ 请先初始化Embedding模型")
            return False

        try:
            # 默认索引参数（HNSW）
            if index_params is None:
                index_params = {
                    "index_type": DEFAULT_INDEX_TYPE,
                    "metric_type": DEFAULT_METRIC_TYPE,
                    "params": {
                        "M": DEFAULT_HNSW_M,
                        "efConstruction": DEFAULT_HNSW_EF_CONSTRUCTION,
                    },
                }

            # 默认搜索参数
            if search_params is None:
                search_params = {
                    "metric_type": DEFAULT_METRIC_TYPE,
                    "params": {
                        "ef": DEFAULT_HNSW_EF,
                    },
                }

            # 构建连接参数
            connection_args = {
                "uri": f"http://{self.host}:{self.port}",
                "token": "root:Milvus",
                "db_name": self.db_name,
            }

            # 创建向量存储
            kwargs = {
                "embedding_function": self.embeddings,
                "connection_args": connection_args,
                "index_params": index_params,
                "search_params": search_params,
                "consistency_level": "Strong",
                "drop_old": self.drop_old,
            }

            # 如果指定了集合名称，添加到参数中
            if self.collection_name:
                kwargs["collection_name"] = self.collection_name

            self.vector_store = Milvus(**kwargs)

            if self.verbose:
                logger.info("✅ 向量存储创建成功")

            return True
        except Exception as e:
            logger.error(f"❌ 向量存储创建失败: {e}")
            return False

    # ==================== 文档管理 ====================

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        添加文本到向量存储

        Args:
            texts: 文本列表
            metadatas: 元数据列表（可选）

        Returns:
            bool: 添加是否成功
        """
        if not self.vector_store:
            logger.error("❌ 请先创建向量存储")
            return False

        try:
            self.vector_store.add_texts(texts, metadatas=metadatas)

            if self.verbose:
                logger.info(f"✅ 成功添加 {len(texts)} 条文档")

            return True
        except Exception as e:
            logger.error(f"❌ 添加文档失败: {e}")
            return False

    def add_documents(self, documents: List[Any]) -> bool:
        """
        添加Document对象到向量存储（推荐用于RAG场景）

        Args:
            documents: Document对象列表（来自LangChain的文档加载器）

        Returns:
            bool: 添加是否成功
        """
        if not self.vector_store:
            logger.error("❌ 请先创建向量存储")
            return False

        try:
            self.vector_store.add_documents(documents)

            if self.verbose:
                logger.info(f"✅ 成功添加 {len(documents)} 条文档")
                # 统计文档来源
                sources = set()
                for doc in documents:
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        sources.add(doc.metadata["source"])
                if sources:
                    logger.info(f"文档来源: {len(sources)} 个文件")

            return True
        except Exception as e:
            logger.error(f"❌ 添加文档失败: {e}")
            return False

    # ==================== 检索和搜索 ====================

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_K,
        filter: Optional[Dict[str, Any]] = None,
    ) -> CompiledStateGraph:
        """
        相似度搜索并创建RAG工作流

        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 过滤条件（可选）

        Returns:
            CompiledStateGraph: 编译后的工作流图，失败返回None
        """
        if not self.vector_store:
            logger.error("❌ 请先创建向量存储")
            return None

        try:
            # results = self.vector_store.similarity_search(query, k=k, expr=filter)
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            self.retriever_tool = create_retriever_tool(
                retriever,
                RETRIEVER_TOOL_NAME,
                RETRIEVER_TOOL_DESCRIPTION,
            )

            graph = self._build_graph()

            return graph
        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            return None

    def mmr_search(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_K,
        fetch_k: int = DEFAULT_MMR_FETCH_K,
        lambda_mult: float = DEFAULT_MMR_LAMBDA,
    ) -> List[Any]:
        """
        最大边际相关性搜索（MMR），减少冗余、提升多样性

        Args:
            query: 查询文本
            k: 返回结果数量
            fetch_k: 初步检索的文档数量
            lambda_mult: 多样性参数（0-1），越小越多样

        Returns:
            List: 搜索结果列表
        """
        if not self.vector_store:
            logger.error("❌ 请先创建向量存储")
            return []

        try:
            results = self.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )
            # retriever = self.vector_store.as_retriever(
            #     search_type="mmr",
            #     search_kwargs={
            #         "k": k,
            #         "fetch_k": fetch_k,
            #         "lambda_mult": lambda_mult,
            #     },
            # )
            # results = retriever.invoke(query)

            if self.verbose:
                logger.info(f"MMR搜索找到 {len(results)} 条多样化文档")

            return results
        except Exception as e:
            logger.error(f"❌ MMR搜索失败: {e}")
            return []

    # ==================== 集合管理 ====================

    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """
        获取集合信息

        Returns:
            Dict: 集合信息字典，失败返回None
        """
        try:
            collections = utility.list_collections()

            if not collections:
                logger.info("当前数据库没有集合")
                return None

            # 获取第一个集合的信息（或指定的集合）
            target_collection = (
                self.collection_name if self.collection_name else collections[0]
            )

            if target_collection not in collections:
                logger.warning(f"集合 '{target_collection}' 不存在")
                return None

            collection = Collection(name=target_collection)
            collection.load()

            info = {
                "name": target_collection,
                "is_empty": collection.is_empty,
                "description": collection.description,
            }

            # 获取向量维度
            for field in collection.schema.fields:
                if field.dtype.name == "FLOAT_VECTOR":
                    info["vector_dim"] = field.params.get("dim")
                    break

            # 获取索引信息
            indexes = collection.indexes
            info["indexes"] = [
                {"field": idx.field_name, "params": idx.params} for idx in indexes
            ]

            return info
        except Exception as e:
            logger.error(f"❌ 获取集合信息失败: {e}")
            return None

    def drop_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        删除集合

        Args:
            collection_name: 集合名称（可选，默认使用当前集合）

        Returns:
            bool: 删除是否成功
        """
        try:
            target_name = collection_name or self.collection_name

            if not target_name:
                logger.error("❌ 未指定集合名称")
                return False

            collections = utility.list_collections()

            if target_name not in collections:
                logger.warning(f"集合 '{target_name}' 不存在")
                return False

            collection = Collection(name=target_name)
            collection.drop()

            if self.verbose:
                logger.info(f"✅ 成功删除集合 '{target_name}'")

            return True
        except Exception as e:
            logger.error(f"❌ 删除集合失败: {e}")
            return False

    # ==================== RAG 工作流 ====================

    def generate_query_or_respond(self, state: MessagesState) -> Dict[str, List]:
        """
        生成查询或直接响应

        优先调用检索工具检索；若未检索到相关结果则返回'未找到相关内容'的回复。

        Args:
            state: 消息状态

        Returns:
            Dict: 更新后的消息字典
        """
        response = self.llm.bind_tools([self.retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}

    def grade_documents(
        self,
        state: MessagesState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        """
        评估检索文档的相关性

        判断检索到的文档是否与该问题相关。

        Args:
            state: 消息状态

        Returns:
            str: 下一步操作（"generate_answer" 或 "rewrite_question"）
        """
        question = state["messages"][0].content
        context = state["messages"][-1].content

        prompt = GRADE_PROMPT_TEMPLATE.format(question=question, context=context)
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        # 提取响应内容并转换为小写进行判断
        score = response.content.strip().lower()

        if self.verbose:
            logger.info(f"📊 文档相关性评分: {score}")

        if "yes" in score:
            return "generate_answer"
        else:
            return "rewrite_question"

    def rewrite_question(self, state: MessagesState) -> Dict[str, List]:
        """
        重写用户问题

        对原始问题进行改写以提高检索效果。

        Args:
            state: 消息状态

        Returns:
            Dict: 包含重写后问题的消息字典
        """
        messages = state["messages"]
        question = messages[0].content
        prompt = REWRITE_PROMPT_TEMPLATE.format(question=question)
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}

    def generate_answer(self, state: MessagesState) -> Dict[str, List]:
        """
        生成最终答案

        基于检索到的上下文生成回答。

        Args:
            state: 消息状态

        Returns:
            Dict: 包含生成答案的消息字典
        """
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = GENERATE_PROMPT_TEMPLATE.format(question=question, context=context)
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    def _build_graph(
        self, save_path: Optional[str] = DEFAULT_GRAPH_PATH
    ) -> CompiledStateGraph:
        """
        构建RAG工作流图（私有方法）

        创建包含检索、评估、重写和答案生成的完整工作流。

        Args:
            save_path: 工作流图保存路径（可选）

        Returns:
            CompiledStateGraph: 编译后的工作流图
        """
        workflow = StateGraph(MessagesState)

        # 定义工作流中会切换的节点
        workflow.add_node(self.generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node(self.rewrite_question)
        workflow.add_node(self.generate_answer)

        # 设置起始边
        workflow.add_edge(START, "generate_query_or_respond")

        # 判断是否需要检索
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,  # 判断LLM的决策（调用工具还是直接回复）
            {
                "tools": "retrieve",  # 需要检索
                END: END,  # 直接结束
            },
        )

        # 检索后评估文档相关性
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents,  # 评估文档相关性
        )

        # 设置其他边
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")

        # 编译工作流
        graph = workflow.compile()

        # 尝试保存工作流图
        self._save_graph_image(graph, save_path)

        return graph

    def _save_graph_image(
        self, graph: CompiledStateGraph, save_path: Optional[str] = None
    ) -> None:
        """
        保存工作流图为图片（私有方法）

        Args:
            graph: 编译后的工作流图
            save_path: 保存路径（可选）
        """
        try:
            png_data = graph.get_graph().draw_mermaid_png()

            if save_path:
                with open(save_path, "wb") as f:
                    f.write(png_data)
                if self.verbose:
                    logger.info(f"✅ 工作流图已保存到: {save_path}")
            else:
                # 尝试在 Jupyter 中显示
                try:
                    display(Image(png_data))
                except NameError:
                    # 如果不在 Jupyter 环境中，保存到默认位置
                    default_path = DEFAULT_GRAPH_PATH
                    with open(default_path, "wb") as f:
                        f.write(png_data)
                    if self.verbose:
                        logger.info(f"📊 工作流图已保存到: {default_path}")
        except Exception as e:
            logger.warning(f"⚠️ 无法生成工作流图: {e}")

    # ==================== 初始化和资源管理 ====================

    def initialize(self) -> bool:
        """
        一键初始化所有组件

        执行以下初始化步骤：
        1. 连接Milvus数据库
        2. 设置/创建数据库
        3. 初始化Embedding模型
        4. 创建向量存储
        5. 初始化LLM响应模型

        Returns:
            bool: 初始化是否成功
        """
        steps = [
            ("连接数据库", self.connect),
            ("设置数据库", self.setup_database),
            ("初始化Embedding模型", self.init_embeddings),
            ("创建向量存储", self.create_vector_store),
            ("初始化响应模型", self.init_response_model),
        ]

        for step_name, step_func in steps:
            if not step_func():
                logger.error(f"❌ 初始化失败于步骤: {step_name}")
                return False

        if self.verbose:
            logger.info("🎉 Milvus工具类初始化完成")

        return True
