import os
import pickle
import docx
import faiss
import numpy as np
import ollama
import openpyxl
from pypdf import PdfReader

# Укажите путь к новому документу, который нужно добавить
NEW_FILE_PATH = "./new_recommendation.pdf"


# --- ТЕ ЖЕ ФУНКЦИИ ЧТЕНИЯ И НАРЕЗКИ (для автономности скрипта) ---
def read_pdf(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Ошибка чтения PDF: {e}")
    return text


def read_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Ошибка чтения DOCX: {e}")
    return text


def read_excel(file_path):
    text = ""
    try:
        wb = openpyxl.load_workbook(file_path, data_only=True)
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            for row in ws.iter_rows(values_only=True):
                row_text = " ".join(
                    [str(cell) for cell in row if cell is not None]
                )
                if row_text.strip():
                    text += row_text + "\n"
    except Exception as e:
        print(f"Ошибка чтения Excel: {e}")
    return text


def split_text(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# --- ОСНОВНОЙ ПРОЦЕСС ДОБАВЛЕНИЯ ---

if not os.path.exists(NEW_FILE_PATH):
    print(f"❌ Файл {NEW_FILE_PATH} не найден!")
    exit()

# 1. Загружаем ТЕКУЩУЮ базу знаний с диска
print("📂 Загружаем существующую базу FAISS и метаданные...")
index = faiss.read_index("cis_faiss.index")
with open("cis_metadata.pkl", "rb") as f:
    documents_metadata = pickle.load(f)

# Определяем текущий максимальный ID в базе, чтобы продолжить нумерацию
current_max_id = max(documents_metadata.keys()) if documents_metadata else -1
print(
    f"ℹ️ Сейчас в базе фрагментов: {index.ntotal} (Максимальный ID: {current_max_id})"
)

# 2. Читаем новый файл
file_name = os.path.basename(NEW_FILE_PATH)
ext = file_name.lower()
file_text = ""

if ext.endswith(".pdf"):
    file_text = read_pdf(NEW_FILE_PATH)
elif ext.endswith(".docx") or ext.endswith(".doc"):
    file_text = read_docx(NEW_FILE_PATH)
elif ext.endswith(".xlsx") or ext.endswith(".xls"):
    file_text = read_excel(NEW_FILE_PATH)

if not file_text.strip():
    print("❌ Не удалось извлечь текст из файла или файл пустой.")
    exit()

# 3. Нарезаем текст на чанки
new_chunks = split_text(file_text)
print(f"📄 Файл {file_name} успешно прочитан. Создано фрагментов: {len(new_chunks)}")

# 4. Генерируем векторы и поочередно добавляем их прямо в текущую базу
print("🔄 Создаем эмбеддинги и обновляем индексы...")
new_embeddings = []

for chunk in new_chunks:
    response = ollama.embed(model="nomic-embed-text", input=chunk)
    new_embeddings.append(response["embeddings"])

# Переводим в массив NumPy
new_embeddings_array = np.array(new_embeddings, dtype=np.float32)

# ВАЖНО: Просто добавляем новые векторы в существующий индекс FAISS! [1.1]
index.add(new_embeddings_array)

# 5. Дописываем новые тексты в словарь метаданных
for i, chunk in enumerate(new_chunks):
    next_id = current_max_id + 1 + i
    documents_metadata[next_id] = {"text": chunk, "source": file_name}

# 6. Перезаписываем обновленные файлы на диск
faiss.write_index(index, "cis_faiss.index")
with open("cis_metadata.pkl", "wb") as f:
    pickle.dump(documents_metadata, f)

print(
    f"\n✅ УСПЕХ! Новый документ успешно интегрирован без полной переиндексации."
)
print(f"📈 Новое общее количество фрагментов в базе: {index.ntotal}")
