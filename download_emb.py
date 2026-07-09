from sentence_transformers import SentenceTransformer

# 使用国内HF镜像托管的模型，下载稳定
model = SentenceTransformer("shibing624/all-MiniLM-L6-v2")
print("✅ 向量模型下载+本地缓存完成，可正常运行RAG程序")
input("按回车键关闭窗口...")