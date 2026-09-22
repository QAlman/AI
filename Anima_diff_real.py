import torch
from PIL import Image
from diffusers import AnimateDiffSDXLPipeline, MotionAdapter
from diffusers.utils import export_to_gif

# --- НАСТРОЙКА ПУТЕЙ К ЛОКАЛЬНЫМ ФАЙЛАМ ---
# Путь к файлу, который вы только что скачали (949 MB)
PATH_TO_MOTION_ADAPTER = "./models/animatediff_motion_sdxl_beta.safetensors" 
# Путь к большой модели SDXL или Pony, которую вам нужно скачать (6-7 GB)
PATH_TO_SDXL_CHECKPOINT = "./models/pony_diffusion_v6_xl.safetensors" 

# 1. Загружаем ваше оригинальное фото
# Для SDXL стандартное разрешение выше — лучше всего использовать квадрат 1024x1024
image_path = "./my_photo.jpg"
init_image = Image.open(image_path).convert("RGB").resize((1024, 1024))

# Делаем из фото массив из 16 одинаковых кадров для будущей анимации
init_frames = [init_image] * 16

print("1. Загрузка локального SDXL Motion Adapter...")
adapter = MotionAdapter.from_single_file(
    PATH_TO_MOTION_ADAPTER, 
    torch_dtype=torch.float16
)

print("2. Загрузка основной чекпоинт-модели (SDXL/Pony)...")
pipe = AnimateDiffSDXLPipeline.from_single_file(
    PATH_TO_SDXL_CHECKPOINT,
    motion_adapter=adapter,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

print("3. Подключение IP-Adapter для SDXL (сохранение лица)...")
# Автоматически загружаем адаптер лица для архитектуры SDXL
pipe.load_ip_adapter(
    "h94/IP-Adapter", 
    subfolder="sdxl_models", 
    weight_name="ip-adapter-plus-face_sdxl_vit-h.bin"
)
# Устанавливаем силу удержания лица (0.7-0.8 оптимально)
pipe.set_ip_adapter_scale(0.75)

# Включаем важные оптимизации для экономии памяти на Tesla V100 16GB / Mac 32GB
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

print("4. Запуск генерации видео...")
# Если используете Pony, добавьте в начало промпта: score_9, score_8_up, masterpiece
prompt = "cinematic motion, hair blowing in the wind, soft studio lighting, highly detailed, 4k"
negative_prompt = "bad quality, low quality, deformed face, blurry, static"

output = pipe(
    video=init_frames,            # Передаем массив кадров из фото
    ip_adapter_image=init_image,  # Передаем фото как визуальный промпт для лица
    prompt=prompt,
    negative_prompt=negative_prompt,
    strength=0.65,                # Сила изменения (0.65 — идеальный баланс движения и схожести)
    guidance_scale=6.5,
    num_inference_steps=25,
    generator=torch.manual_seed(42)
)

print("5. Сохранение результата...")
output_path = "animated_sdxl_face.gif"
export_to_gif(output.frames, output_path, fps=8)
print(f"Успешно! Анимация сохранена в файл: {output_path}")

#https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/tree/main
#https://civitai.com/models/133005/juggernaut-xl
