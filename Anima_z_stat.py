import torch
from PIL import Image
from diffusers import ZImagePipeline

# 1. Путь к вашему скачанному файлу IntoRealism (5.73 GB)
PATH_TO_ZIT_MODEL = "./models/IntoRealismZIT9FP8.safetensors"

# 2. Загружаем ваше оригинальное локальное фото
# Z-Image любит разрешения в районе 896x1152 или 832x1216
init_image = Image.open("./my_photo.jpg").convert("RGB").resize((832, 1216))

print("Загрузка локальной модели IntoRealism ZIT V9.0...")
# Загружаем пайплайн напрямую из одного .safetensors файла
pipe = ZImagePipeline.from_single_file(
    PATH_TO_ZIT_MODEL,
    torch_dtype=torch.float16,  # Адаптируем под Tesla V100 / Mac
    use_safetensors=True
)

# Включаем оптимизацию памяти для стабильной работы
pipe.enable_model_cpu_offload()

print("Процесс обработки фото запущен...")
# Пишем промпт на естественном языке (без тегов вроде score_9)
# Описываем изменения, которые хотим увидеть на человеке с фото
prompt = "A high-tech cyberpunk leather jacket with neon glowing blue lines, photorealistic, cinematic studio lighting, highly detailed skin texture, 8k"

# Запускаем генерацию по рекомендациям автора модели (enzino)
output = pipe(
    prompt=prompt,
    image=init_image,          # Передаем наше исходное фото
    num_inference_steps=8,     # Автор рекомендует от 6 до 10 шагов для ZIT
    guidance_scale=1.0,        # Автор жестко рекомендует CFG = 1 для ZIT режима!
    strength=0.6,              # Сила изменения фото (0.5 - аккуратно, 0.7 - сильно)
    generator=torch.manual_seed(123)
)

# 3. Сохраняем измененную фотографию
output.images[0].save("intorealism_output.png")
print("Готово! Результат сохранен в файл intorealism_output.png")
