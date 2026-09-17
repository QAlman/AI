import pickle
import faiss
import numpy as np
import ollama

# 1. Загружаем нашу базу FAISS и тексты, которые мы сохранили ранее
index = faiss.read_index("cis_faiss.index")
with open("cis_metadata.pkl", "rb") as f:
    documents_metadata = pickle.load(f)

# 2. Вопрос пользователя (например, по логам или безопасности)
user_query = "Help me with asset management according to CIS standards."
print(f"❓ Вопрос пользователя: {user_query}\n")

# 3. Извлекаем контекст из базы (RETRIEVAL)
print("🔍 Ищем подсказки в базе данных FAISS...")
# Делаем вектор запроса через маленькую модель nomic
response_embed = ollama.embed(model="nomic-embed-text", input=user_query)
query_embedding = np.array(response_embed["embeddings"], dtype=np.float32)

# Ищем 1 самый похожий документ
distances, indices = index.search(query_embedding, k=1)
best_match_idx = indices[0][0]

if best_match_idx != -1:
    context_text = documents_metadata[best_match_idx]["text"]
    print(f"📌 Найдено совпадение в базе: {context_text}\n")
else:
    context_text = "No specific CIS control found."
    print("⚠️ Совпадений в базе не найдено.\n")

# 4. Формируем Промпт (инструкцию) для большой модели (AUGMENTED)
# Мы заставляем модель отвечать СТРОГО по тексту из нашей базы данных
prompt = f"""You are a professional cybersecurity expert. 
Answer the user's question using ONLY the provided CIS Control Context. 
If the context doesn't contain the answer, say that you don't know.

CIS Control Context:
{context_text}

User Question:
{user_query}

Expert Answer:"""

# 5. Генерируем ответ через большую модель ornith (GENERATION)
print(f"🤖 Модель ornith-1.5:9b генерирует ответ...")
output = ollama.generate(
    model="ornith-1.5:9b",
    prompt=prompt
)

print("\n✨ ИТОГОВЫЙ ОТВЕТ СИСТЕМЫ:")
print(output["response"])
