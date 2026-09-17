import os
import pickle
import faiss
import numpy as np
import ollama

# Исходный документ CIS
document_text = "CIS Control 1: Inventory and Control of Enterprise Assets. Maintain an accurate and up-to-date inventory of all enterprise assets."

print("🔄 Шаг 1: Получаем вектор от Ollama...")
response = ollama.embed(
    model="nomic-embed-text",
    input=document_text
)
# Превращаем вектор в массив NumPy с типом float32 (FAISS требует именно этот тип)
embedding = np.array(response["embeddings"], dtype=np.float32)

# Узнаем размерность вектора (для nomic-embed-text это обычно 768)
dimension = embedding.shape[1]

print("💾 Шаг 2: Создаем базу данных FAISS...")
# Создаем индекс FAISS для поиска по косинусному/евклидову расстоянию
index = faiss.IndexFlatL2(dimension)

# Добавляем наш вектор в базу данных
index.add(embedding)

# Т.к. FAISS хранит только цифры, сам текст мы сохраним рядышком в простой список
documents_metadata = {
    0: {
        "text": document_text,
        "source": "cis_handbook",
        "chapter": "1"
    }
}

print("📁 Шаг 3: Сохраняем базу данных на диск...")
# Сохраняем векторный индекс
faiss.write_index(index, "cis_faiss.index")

# Сохраняем тексты документов в файл, чтобы потом сопоставить вектор с текстом
with open("cis_metadata.pkl", "wb") as f:
    pickle.dump(documents_metadata, f)

print("✅ ВСЁ ГОТОВО! База успешно создана и сохранена без ошибок!")
