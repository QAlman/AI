import os
import base64
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_SHARE"] = "False"
os.environ["GRADIO_DISABLE_UPDATE_CHECK"] = "True"
import gradio as gr
from llama_cpp import Llama
# Исправленный импорт стандартного обработчика для мультимодальных моделей
from llama_cpp.llama_chat_format import Llava15ChatHandler

# 1. Пути к файлам модели в вашей папке
model_path = r"C:\MODEL\Qwen3.5-4B-Q4_K_M.gguf"
clip_path = r"C:\MODEL\mmproj-Qwen3.5-4B-BF16.gguf"

if not os.path.exists(model_path) or not os.path.exists(clip_path):
    raise FileNotFoundError(
        f"Убедитесь, что файлы '{model_path}' и '{clip_path}' лежат в папке со скриптом!"
    )

print("Загрузка мультимодального проектора (mmproj)...")
# Используем Llava15ChatHandler — универсальный коннектор для mmproj
chat_handler = Llava15ChatHandler(clip_model_path=clip_path)

print("Загрузка основной модели Gemma-4 (может занять до 1-2 минут)...")
llm = Llama(
    model_path=model_path,
    n_ctx=4096,  # 4k контекста хватает для картинки и текстового вопроса
    n_gpu_layers=12,  # Использовать видеокарту. Если только CPU — измените на 0
    chat_handler=chat_handler,  # Передаем наш проектор
    verbose=False
)




# 2. Обработка входящих данных из интерфейса Gradio
import base64

def predict(user_input, history):
    text = user_input.get("text", "") or ""
    files = user_input.get("files", []) or []

    messages = [
        {"role": "system",
         "content": "Ты — полезный мультимодальный ИИ-ассистент Gemma-4. Отвечай на русском языке."}
    ]

    # --- История: формат messages (Gradio 6) ---
    for turn in history:
        if isinstance(turn, dict):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            u, a = turn
            if u:
                messages.append({"role": "user", "content": u})
            if a:
                messages.append({"role": "assistant", "content": a})

    # --- Текущий ввод ---
    current_content = []
    if text.strip():
        current_content.append({"type": "text", "text": text})

    for file_path in files:
        if isinstance(file_path, str) and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            current_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

    if not current_content:
        yield history + [{"role": "assistant", "content": "Вы отправили пустое сообщение."}]
        return

    messages.append({"role": "user", "content": current_content})

    # --- Стриминг ---
    response_stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
        stream=True
    )

    partial_message = ""
    user_turn = {"role": "user", "content": text}
    for chunk in response_stream:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        token = delta.get("content")
        if token:
            partial_message += token
            yield history + [
                user_turn,
                {"role": "assistant", "content": partial_message}
            ]
    # После завершения стриминга обновляем основную историю
    history.append({"role": "assistant", "content": partial_message})

# 3. Веб-интерфейс
with gr.Blocks(title="Мультимодальная Gemma-4") as demo:
    gr.Markdown("# 👁️ Локальная Мультимодальная Gemma-4 E4B")
    gr.Markdown("Загружайте графики, диаграммы или фотографии и задавайте по ним вопросы.")

    chatbot = gr.Chatbot(label="Чат")

    with gr.Row():
        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_types=["image"],
            placeholder="Введите вопрос или прикрепите изображение через кнопку + ...",
            show_label=False,
            scale=8
        )

    chat_input.submit(
        fn=predict,
        inputs=[chat_input, chatbot],
        outputs=chatbot
    ).then(
        fn=lambda: None,
        inputs=None,
        outputs=chat_input
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
