from milvusRAG.index import RAGBuilder


def main():
    """主函数：构建RAG索引并测试搜索"""
    # 1. 创建RAG构建器
    builder = RAGBuilder()

    # 2. 构建RAG索引
    milvus = builder.build_rag_index(
        data_dir="./data",  # 数据目录
        db_name="rag_database",  # Milvus数据库名称
        collection_name="my_documents",  # 集合名称
        chunk_size=500,  # 文档分块大小
        chunk_overlap=50,  # 分块重叠大小
        drop_old=False,  # 是否删除已存在的集合（True=重建索引）
        verbose=True,  # 是否显示详细日志
    )

    if not milvus:
        print("❌ RAG索引构建失败")
        return

    # 3. 测试搜索功能
    print("\n" + "=" * 60)
    print("测试RAG搜索")
    print("=" * 60)

    query = "如何使用无头模式运行？"
    print(f"\n查询: {query}")

    # 使用相似性搜索创建工作流图
    graph = milvus.similarity_search(query, k=3)

    # 流式输出结果
    for chunk in graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    ):
        for node, update in chunk.items():
            print(f"\n节点: {node}")
            update["messages"][-1].pretty_print()

    # 4. 关闭连接
    milvus.close()
    print("\n✅ 完成")


if __name__ == "__main__":
    main()
