import torch
from PIL import Image
from diffusers import ZImagePipeline # Импортируем родной пайплайн Z-Image

# 1. Загружаем фотографию (Z-Image отлично работает с разрешением 1024x1024)
init_image = Image.open("./my_photo.jpg").convert("RGB").resize((1024, 1024))

print("Загрузка модели Z-Image-Turbo...")
# Модель скачается автоматически из официального репозитория Alibaba
model_id = "Tongyi-MAI/Z-Image-Turbo"
pipe = ZImagePipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,  # Адаптируем под Tesla V100 (вместо bfloat16)
    device_map="cuda"
)

# Оптимизация памяти (для версий карт 16 ГБ)
pipe.enable_model_cpu_offload()

print("Генерация...")
# Пишем промпт на естественном языке. Никаких "score_9" или "masterpiece" не нужно!
prompt = "A high-tech cyberpunk jacket on the person, neon glowing lines, highly detailed, photorealistic, 8k"

# Запускаем пайплайн (для Turbo-модели достаточно всего 8 шагов)
output = pipe(
    prompt=prompt,
    image=init_image,          # Передаем исходное фото для изменения/стилизации
    num_inference_steps=8,     # Всего 8 шагов вместо 25-30 у SDXL!
    guidance_scale=0.0,        # Для Turbo-дистилляции CFG отключают
    strength=0.6,              # Сила изменения оригинального фото
)

# Сохраняем результат
output.images[0].save("z_image_output.png")
print("Готово! Картинка успешно обновлена.")


#https://civitai.com/models/1609320/intorealism?modelVersionId=3258780
