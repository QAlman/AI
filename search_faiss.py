import pickle
import faiss
import numpy as np
import ollama

# 1. Загружаем базу с диска
index = faiss.read_index("cis_faiss.index")
with open("cis_metadata.pkl", "rb") as f:
    documents_metadata = pickle.load(f)

# 2. Наш поисковый запрос
query = "How to manage asset inventory?"

# 3. Переводим запрос в вектор
response = ollama.embed(model="nomic-embed-text", input=query)
query_embedding = np.array(response["embeddings"], dtype=np.float32)

# 4. Ищем в FAISS 1 самый похожий документ
# k=1 означает "верни 1 лучший результат"
distances, indices = index.search(query_embedding, k=1)

# 5. Выводим результат
best_match_idx = indices[0][0]
if best_match_idx != -1:
    found_doc = documents_metadata[best_match_idx]
    print("\n🔎 Результат поиска:")
    print(f"Текст: {found_doc['text']}")
    print(f"Источник: {found_doc['source']} (Глава {found_doc['chapter']})")
    print(f"Дистанция схожести: {distances[0][0]:.4f} (чем меньше, тем ближе смысл)")
else:
    print("Ничего не найдено.")
