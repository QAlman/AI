import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline

def run_img2img(
    model_path: str, 
    input_image_path: str, 
    output_image_path: str, 
    prompt: str, 
    negative_prompt: str = "", 
    denoising_strength: float = 0.4, 
    steps: int = 30, 
    cfg_scale: float = 7.0
):
    print("⏳ Загрузка пайплайна и модели...")
    
    # Загружаем SDXL Img2Img пайплайн напрямую из локального .safetensors файла
    pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # Переносим вычисления на видеокарту (CUDA)
    pipeline.to("cuda")
    
    # Включаем оптимизацию памяти для экономии VRAM
    pipeline.enable_model_cpu_offload() 
    
    print("📸 Подготовка изображения...")
    # Открываем картинку и приводим к RGB, масштабируем под стандарты SDXL (кратно 64)
    init_image = Image.open(input_image_path).convert("RGB")
    init_image = init_image.resize((1024, 1024))  # SDXL лучше всего работает с 1024x1024
    
    print(f"🚀 Генерация (Denoising strength: {denoising_strength})...")
    
    # Запуск генерации
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=denoising_strength,     # Чем меньше, тем ближе к оригиналу
        guidance_scale=cfg_scale,        # Соответствие промпту (CFG)
        num_inference_steps=steps,       # Количество шагов
    ).images[0]
    
    # Сохраняем результат
    image.save(output_image_path)
    print(f"✅ Готово! Изображение сохранено в: {output_image_path}")

# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---
if __name__ == "__main__":
    
    # 1. Укажите пути к вашим файлам
    # Замените пути на реальные папки, куда вы скачали модели с Civitai
    PATH_TO_MODEL = r"C:\AI\models\juggernautXL_v9.safetensors" 
    # Для Pony переключите путь: r"C:\AI\models\ponyDiffusionV6_v6.safetensors"
    
    INPUT_IMG = "my_photo.jpg"
    OUTPUT_IMG = "result_photo.jpg"
    
    # 2. Настройки промптов
    # ВАЖНО: Для Pony V6 обязательно добавляйте в начало промпта: "score_9, score_8_up, score_7_up, ..."
    PROMPT = "a professional cinematic portrait of a man, wearing a black leather jacket, highly detailed, 8k resolution"
    NEG_PROMPT = "deformed, bad anatomy, disfigured, blurry, low quality"
    
    # Настройка схожести: 
    # 0.3 - очень похоже на оригинал, легкие правки
    # 0.5 - баланс между изменениями и сохранением лица/композиции
    # 0.7 - сильные изменения, оригинал останется лишь блеклым контуром
    STRENGTH = 0.4 
    
    # Проверим, что видеокарта доступна
    if not torch.cuda.is_available():
        print("❌ Ошибка: CUDA (Nvidia GPU) не обнаружена! Скрипт не сможет работать эффективно.")
    else:
        run_img2img(
            model_path=PATH_TO_MODEL,
            input_image_path=INPUT_IMG,
            output_image_path=OUTPUT_IMG,
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            denoising_strength=STRENGTH,
            steps=35,
            cfg_scale=6.5
        )
