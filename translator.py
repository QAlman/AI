import os
import gradio as gr
from llama_cpp import Llama
from datetime import datetime

# === НАСТРОЙКИ ПУТЕЙ ===
MODEL_PATH = r"E:\AI\MODEL\2\gemma-4-E2B_q4_0-it.gguf"  # Ваша модель
INPUT_FOLDER = r"D:\AI\TRS\input"  # Папка для чтения файлов
OUTPUT_FOLDER = r"D:\AI\TRS\output"  # Папка для сохранения результатов

# Автоматически создаем папки, если их нет
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# 1. Получаем текущую дату и время
start_time = datetime.now()

# Инициализируем модель (добавьте n_threads для вашего CPU, например 4, 6 или 8)
print("Загрузка модели...")
llm = Llama(
    model_path=MODEL_PATH,

    n_ctx=8096,  # Увеличиваем контекст для длинных текстов
    n_threads=6,  # Укажите количество ядер вашего процессора
    n_gpu_layers=-1, # -1 Измените на 0, если запускаете строго на процессоре
    chat_format="gemma",
    verbose=False,
)


# Функция ядра перевода
def translate_core(text, direction):
    if direction == "English -> Russian":
        src, target = "English", "Russian"
    else:
        src, target = "Russian", "English"

    prompt = (
        "<|im_start|>system\n"
        f"You are a professional translator. Translate the user's text from {src} to {target} accurately and naturally. "
        "Do not write any explanations or introductory text. Respond ONLY with the translation.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{text}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    output = llm(
        prompt,
        max_tokens=8048,  # Запас токенов на перевод глав/страниц
        temperature=0.05,
        stop=["<|im_end|>"] # stop=["<|im_end|>", "<|endoftext|>"]
    )
    return output['choices'][0]['text'].strip()


# 1. Функция работы через текстовое поле веб-интерфейса
def web_translate(text, direction):
    if not text.strip():
        return "Пожалуйста, введите текст."
    return translate_core(text, direction)


# 2 и 3. Функция для работы с файлами
def file_translate(file_name, direction):
    if not file_name:
        return "Файл не выбран.", ""

    # Чтение файла из папки INPUT_FOLDER
    input_path = os.path.join(INPUT_FOLDER, file_name)
    if not os.path.exists(input_path):
        return f"Файл {file_name} не найден в папке input!", ""

    with open(input_path, "r", encoding="utf-8") as f:
        text_to_translate = f.read()

    # Перевод
    translated_text = translate_core(text_to_translate, direction)

    # 2. Форматируем по вашему шаблону: год (%Y) день (%d) месяц (%m) минута (%M)
    # Пример вывода: 2026_20_09_12
    date_str = start_time.strftime("%Y_%m_%d_%H_%M")

    # Сохранение файла в папку OUTPUT_FOLDER
    output_file_name = f"{date_str}_{direction.replace(' -> ', '_')}_{file_name}"
    output_path = os.path.join(OUTPUT_FOLDER, output_file_name)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated_text)



    return f"Файл успешно прочитан из папки 'input'.\nРезультат сохранен как: {output_file_name}", translated_text


# Функция для обновления списка файлов в интерфейсе
def get_file_list():
    files = [f for f in os.listdir(INPUT_FOLDER) if os.path.isfile(os.path.join(INPUT_FOLDER, f))]
    return files if files else ["Папка 'input' пуста. Добавьте .txt файлы."]


# === СБОРКА WEB ИНТЕРФЕЙСА (GRADIO) ===
with gr.Blocks(title="Локальный Переводчик ИИ") as demo:
    gr.Markdown("# 🤖 Локальный ИИ-переводчик (Qwen GGUF)")

    # Выбор направления перевода для всех вкладок
    direction_dropdown = gr.Dropdown(
        choices=["English -> Russian", "Russian -> English"],
        value="English -> Russian",
        label="Направление перевода"
    )

    with gr.Tab("Перевод текста"):
        with gr.Row():
            input_text = gr.Textbox(label="Исходный текст", lines=8, placeholder="Введите текст для перевода...")
            output_text = gr.Textbox(label="Перевод", lines=8, interactive=False)
        translate_btn = gr.Button("Перевести текст", variant="primary")
        translate_btn.click(web_translate, inputs=[input_text, direction_dropdown], outputs=output_text)

    with gr.Tab("Перевод файлов из папки"):
        gr.Markdown(
            f"Положите ваши текстовые файлы (`.txt`) в папку: `{INPUT_FOLDER}`\nПереводы сохранятся в папку: `{OUTPUT_FOLDER}`")

        with gr.Row():
            file_dropdown = gr.Dropdown(choices=get_file_list(), label="Выберите файл для перевода")
            refresh_btn = gr.Button("🔄 Обновить список файлов", scale=0)

        file_status = gr.Textbox(label="Статус операции", interactive=False)
        file_preview = gr.Textbox(label="Превью готового перевода", lines=6, interactive=False)

        file_translate_btn = gr.Button("Перевести выбранный файл", variant="primary")

        # Логика кнопок файлов
        refresh_btn.click(fn=lambda: gr.update(choices=get_file_list()), outputs=file_dropdown)
        file_translate_btn.click(file_translate, inputs=[file_dropdown, direction_dropdown],
                                 outputs=[file_status, file_preview])

# Запуск локального сервера
if __name__ == "__main__":
    # Скрипт запустит веб-страницу по адресу http://127.0.0.1:7860
    demo.launch(inbrowser=True)
