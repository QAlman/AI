# -*- coding: utf-8 -*-
"""
RAG-ассистент без Ollama.
Стек:
  - LLM: gemma-3-4b-it (GGUF, Q4_K_M) через llama-cpp-python + Metal (M1)
  - Embeddings: intfloat/multilingual-e5-large через sentence-transformers
  - Vector store: FAISS (IndexFlatIP, нормализованные векторы)
"""

import os
import pickle
import numpy as np
import faiss
from datetime import datetime

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer


# ============================================================
# ПУТИ И НАСТРОЙКИ
# ============================================================
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH      = os.path.join(BASE_DIR, "cis_faiss.index")
METADATA_PATH   = os.path.join(BASE_DIR, "cis_metadata.pkl")
PROMPT_FILE     = os.path.join(BASE_DIR, "prompt.txt")
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")

# Модели
LLM_REPO_ID     = "daniloreddy/gemma-3-4b-it_GGUF"
LLM_FILENAME    = "gemma-3-4b-it_Q4_K_M.gguf"
EMBED_MODEL_ID  = "intfloat/multilingual-e5-large"

# Параметры RAG
K_RESULTS       = 10
MAX_NEW_TOKENS  = 2048
TEMPERATURE     = 0.7
N_CTX           = 4096
N_GPU_LAYERS    = -1   # M1: всё на Metal. Если ошибка памяти — поставь 20


# ============================================================
# 1. ЗАГРУЗКА МОДЕЛЕЙ (один раз при старте)
# ============================================================
print("🔧 Загружаю LLM (gemma-3-4b-it Q4_K_M)...")
llm = Llama.from_pretrained(
    repo_id=LLM_REPO_ID,
    filename=LLM_FILENAME,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False,
)

print("🔧 Загружаю эмбеддер (multilingual-e5-large)...")
embedder = SentenceTransformer(EMBED_MODEL_ID)


# ============================================================
# 2. ЗАГРУЗКА FAISS И МЕТАДАННЫХ
# ============================================================
print("📂 Загружаю FAISS-индекс и метаданные...")
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    documents_metadata = pickle.load(f)


# ============================================================
# 3. ЧТЕНИЕ И ПОДГОТОВКА ПРОМПТА ИЗ ФАЙЛА
# ============================================================
def read_and_prepare_prompt(file_path: str, default_text: str = "") -> str:
    """
    Читает prompt.txt, нормализует переводы строк, убирает пустые строки
    и лишние пробелы. Возвращает один связный абзац.
    """
    if not os.path.exists(file_path):
        print(f"⚠️  Файл промпта не найден: {file_path}. Использую значение по умолчанию.")
        return default_text

    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    clean_lines = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped:
            clean_lines.append(" ".join(stripped.split()))

    return " ".join(clean_lines)


# ============================================================
# 4. ЗАПРОС ПОЛЬЗОВАТЕЛЯ
# ============================================================
user_query = "Необходимо создать новый документ 'Доступ в интернет' на основе наших документов"
print(f"\n❓ Вопрос пользователя: {user_query}\n")


# ============================================================
# 5. ПОИСК В FAISS
# ============================================================
print("🔍 Ищу совпадения в базе знаний...")

# E5 рекомендует префикс "query: " для запросов
query_vec = embedder.encode(
    [f"query: {user_query}"],
    normalize_embeddings=True,
    convert_to_numpy=True,
).astype(np.float32)

distances, indices = index.search(query_vec, k=K_RESULTS)

context_chunks = []
sources = set()
for idx in indices[0]:
    if idx != -1 and idx in documents_metadata:
        doc = documents_metadata[idx]
        context_chunks.append(doc["text"])
        sources.add(doc["source"])

if context_chunks:
    full_context = "\n---\n".join(context_chunks)
    print(f"📌 Найдено фрагментов: {len(context_chunks)} из файлов: {', '.join(sources)}\n")
else:
    full_context = "Нет релевантных данных в базе."
    print("⚠️  Совпадений в базе не найдено.\n")


# ============================================================
# 6. ФОРМИРОВАНИЕ ИТОГОВОГО ПРОМПТА
# ============================================================
prompt_template = read_and_prepare_prompt(
    PROMPT_FILE,
    default_text="Создай официальный документ на основе предоставленного контекста."
)

final_prompt = f"""{prompt_template}

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{full_context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{user_query}

ОТВЕТ ЭКСПЕРТА:"""


# ============================================================
# 7. ГЕНЕРАЦИЯ ОТВЕТА
# ============================================================
print("🤖 gemma-3-4b-it анализирует контекст и генерирует ответ...")
output = llm(
    prompt=final_prompt,
    max_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    echo=False,
)
response_text = output["choices"][0]["text"].strip()

print("\n✨ ИТОГОВЫЙ ОТВЕТ СИСТЕМЫ:\n")
print(response_text)
print("\n📚 Использованные источники:")
for src in sources:
    print(f" - {src}")


# ============================================================
# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТА
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

now = datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H_%M")
output_filename = os.path.join(OUTPUT_DIR, f"Документ({timestamp}).txt")

with open(output_filename, "w", encoding="utf-8") as f:
    f.write("# КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n")
    f.write(full_context)
    f.write("\n\n")
    f.write("# ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n")
    f.write(user_query)
    f.write("\n\n")
    f.write("# ОТВЕТ ЭКСПЕРТА:\n")
    f.write(response_text)
    f.write("\n\n")
    f.write(f"Дата и время генерации: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Использованные источники: {', '.join(sorted(sources))}\n")

print(f"\n💾 Документ сохранён: {output_filename}")
