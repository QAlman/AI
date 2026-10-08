import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Указываем единую папку, где лежит и .gguf файл, и 4 скачанных файла токенизатора
PATH_TO_FOLDER = r"C:\MODEL\Qwen3.5-4BT"
GGUF_FILE_NAME = "Qwen3.5-4B-Q4_K_M.gguf"

print("--- Загрузка токенизатора И весов строго ИЗ ЛОКАЛЬНОЙ ПАПКИ ---")

# 1. Читаем токенизатор локально (интернет не проверяется)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_FOLDER, local_files_only=True)

# 2. Читаем веса модели локально через PyTorch
model = AutoModelForCausalLM.from_pretrained(
    PATH_TO_FOLDER,
    gguf_file=GGUF_FILE_NAME,
    dtype=torch.float32,  # Защита от системных ошибок на CPU Windows
    device_map="cpu",
    local_files_only=True        # Запрещаем библиотеке выходить в сеть
)

print(" Все компоненты успешно загружены офлайн!")

# Тестовый запуск
prompt = "Привет! Ты меня слышишь?"
inputs = tokenizer(prompt, return_tensors="pt")

# Заставляем PyTorch использовать все доступные ядра процессора для вычислений
torch.set_num_threads(4)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)

print("\n[Ответ модели]:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
