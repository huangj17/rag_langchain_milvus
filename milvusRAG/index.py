"""
RAG索引构建模块

功能：
1. 支持多种文档格式的加载（PDF、Markdown、TXT、CSV、JSON、HTML、DOCX等）
2. 智能文档分割
3. 向量化并存储到Milvus
4. 提供完整的RAG索引构建流程
"""

from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from milvusRAG.milvus_util import MilvusUtil


class RAGBuilder:
    """RAG索引构建类"""

    def get_loader_class(self, file_extension: str):
        """根据扩展名返回合适的 loader 类

        Args:
            file_extension: 文件扩展名（如 '.pdf', '.md'）

        Returns:
            Loader类
        """
        ext = file_extension.lower()

        if ext == ".pdf":
            return PyPDFLoader
        elif ext in [".md", ".txt", ".log"]:
            return TextLoader
        elif ext == ".csv":
            return CSVLoader
        elif ext in [".html", ".htm"]:
            return UnstructuredHTMLLoader
        elif ext in [".docx", ".doc"]:
            return UnstructuredWordDocumentLoader
        else:
            raise ValueError(f"不支持的文件类型: {ext}")

    def load_single_file(self, file_path: str) -> List[Document]:
        """加载单个文件

        Args:
            file_path: 文件路径

        Returns:
            List[Document]: 加载的文档列表
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        try:
            loader_cls = self.get_loader_class(suffix)

            # 对于需要编码参数的TextLoader，添加encoding参数
            if loader_cls == TextLoader:
                loader = loader_cls(str(path), encoding="utf-8")
            else:
                loader = loader_cls(str(path))

            return loader.load()
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            return []

    def smart_directory_loader(self, root_dir: str) -> List[Document]:
        """递归加载一个目录内所有可识别的文件

        Args:
            root_dir: 文档目录路径

        Returns:
            List[Document]: 加载的文档列表
        """
        supported_exts = [
            ".md",
            ".pdf",
            ".txt",
            ".csv",
            ".docx",
            ".html",
            ".htm",
            ".log",
        ]
        docs = []
        root_path = Path(root_dir)

        if not root_path.exists():
            print(f"❌ 目录不存在: {root_dir}")
            return docs

        # 递归遍历目录
        for file_path in root_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                try:
                    file_docs = self.load_single_file(str(file_path))
                    if file_docs:
                        docs.extend(file_docs)
                        print(f"✓ 已加载: {file_path.name}")
                except Exception as e:
                    print(f"✗ 加载 {file_path.name} 时出错: {e}")

        return docs

    def split_docs(
        self,
        docs: List[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_tiktoken: bool = False,
    ) -> List[Document]:
        """分割文档

        Args:
            docs: 文档列表
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            use_tiktoken: 是否使用tiktoken编码器

        Returns:
            List[Document]: 分割后的文档片段
        """
        if use_tiktoken:
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )

        doc_splits = text_splitter.split_documents(docs)
        return doc_splits

    def load_documents(
        self, data_dir: str, chunk_size: int, chunk_overlap: int
    ) -> List[Document]:
        """加载并分割文档

        Args:
            data_dir: 数据目录路径
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小

        Returns:
            List[Document]: 分割后的文档片段列表
        """
        # 1. 加载文档
        print(f"\n2. 加载文档目录: {data_dir}")
        docs = self.smart_directory_loader(data_dir)
        if not docs:
            print("❌ 未找到任何文档")
            return []
        print(f"✅ 成功加载 {len(docs)} 个文档")

        # 2. 分割文档
        print(f"\n3. 分割文档 (chunk_size={chunk_size}, overlap={chunk_overlap})")
        doc_splits = self.split_docs(
            docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        print(f"✅ 分割成 {len(doc_splits)} 个片段")

        return doc_splits

    def build_rag_index(
        self,
        data_dir: str,
        db_name: str = "rag_database",
        collection_name: str = "documents",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        drop_old: bool = False,
        verbose: bool = True,
    ) -> Optional[MilvusUtil]:
        """
        构建RAG索引：加载文档 -> 分割 -> 向量化 -> 存储到Milvus

        Args:
            data_dir: 数据目录路径
            db_name: Milvus数据库名称
            collection_name: 集合名称
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
            drop_old: 是否删除已存在的集合
            verbose: 是否显示详细日志

        Returns:
            MilvusUtil: 初始化完成的Milvus工具实例，失败返回None
        """
        print("=" * 60)
        print("开始构建RAG索引")
        print("=" * 60)

        # 1. 初始化Milvus
        print("\n1. 初始化Milvus向量存储")
        milvus = MilvusUtil(
            db_name=db_name,
            collection_name=collection_name,
            drop_old=drop_old,
            verbose=verbose,
        )

        if not milvus.initialize():
            print("❌ Milvus初始化失败")
            return None

        # 4. 添加文档到向量存储
        print("\n4. 添加文档到向量存储")
        doc_splits = self.load_documents(data_dir, chunk_size, chunk_overlap)
        if not doc_splits:
            print("❌ 未加载到任何文档片段")
            return None

        if not milvus.add_documents(doc_splits):
            print("❌ 添加文档失败")
            return None

        print("\n" + "=" * 60)
        print("🎉 RAG索引构建完成!")
        print("=" * 60)

        return milvus
