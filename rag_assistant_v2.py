import pickle
import faiss
import numpy as np
import ollama

# 1. Загружаем базу FAISS и тексты
index = faiss.read_index("cis_faiss.index")
with open("cis_metadata.pkl", "rb") as f:
    documents_metadata = pickle.load(f)

# 2. Запрос пользователя (можете написать любой вопрос по вашим документам)
user_query = "Необходимо дополнить или создать новый документ 'Требования к инфраструктуре Подрядчика' данными на основе наших документов"
print(f"❓ Вопрос пользователя: {user_query}\n")

# 3. Извлекаем контекст из базы
print("🔍 Ищем совпадения в базе знаний...")
response_embed = ollama.embed(model="nomic-embed-text", input=user_query)
query_embedding = np.array(response_embed["embeddings"], dtype=np.float32)

# Ищем ТОП-4 самых похожих фрагмента (k=4)
k_results = 10
distances, indices = index.search(query_embedding, k=k_results)

# Собираем найденные куски текста в один общий контекст
context_chunks = []
sources = set()

for idx in indices[0]:
    if idx != -1 and idx in documents_metadata:
        doc_info = documents_metadata[idx]
        context_chunks.append(doc_info["text"])
        sources.add(doc_info["source"])

# Соединяем тексты через перенос строки
full_context = "\n---\n".join(context_chunks)

if context_chunks:
    print(f"📌 Найдено фрагментов: {len(context_chunks)} из файлов: {', '.join(sources)}\n")
else:
    full_context = "Нет релевантных данных в базе."
    print("⚠️ Совпадений в базе не найдено.\n")

# 4. Формируем системный промпт (на русском языке, так как модель ornith отлично его понимает)
prompt = f"""На основе существующих в базе политик ИБ и регламентов предоставления доступа к СВП, 
внеси изменения в документ: 'Требования к инфраструктуре Подрядчика'. 
Документ должен содержать:
1. Обязательные требования к АРМ подрядчика.
2. Правила обновления и использования антивирусного ПО.
3. Правила двухфакторной аутентификации и авторизации.
4. Правила на основе рекомендаций ФСТЕК.
5. Правила на основе существующих политик ИБ и управлений доступом.
Пиши официальным техническим языком

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{full_context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{user_query}

ОТВЕТ ЭКСПЕРТА:"""

# 5. Генерируем ответ через большую модель ornith-1.5:9b
print(f"🤖 Модель gemma3:4b анализирует контекст и генерирует ответ...")
output = ollama.generate(
    model="gemma3:4b",
    prompt=prompt,
    options={"temperature": 0.7} # Небольшая температура, чтобы модель меньше фантазировала
)

print("\n✨ ИТОГОВЫЙ ОТВЕТ СИСТЕМЫ:")
print(output["response"])
print("\n📚 Использованные источники:")
for src in sources:
    print(f" - {src}")

# Допишите этот код в самый конец файла rag_assistant_v2.py
output_filename = "Сгенерированный_Документ_1.md"

with open(output_filename, "w", encoding="utf-8") as f:
    f.write(output["response"])

print(f"\n💾 Новый документ успешно сформирован и сохранен в файл: {output_filename}")
