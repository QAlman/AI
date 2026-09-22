import torch
from PIL import Image
from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

# 1. Загружаем ваше оригинальное фото
# Для SD 1.5 лучше всего использовать квадратное разрешение, например 512x512
image_path = "./my_photo.jpg"
init_image = Image.open(image_path).convert("RGB").resize((512, 512))

# Превращаем одно фото в массив из 16 одинаковых кадров (база для анимации)
init_frames = [init_image] * 16

# 2. Инициализируем компоненты AnimateDiff
print("Загрузка модулей движения...")
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-v2", 
    torch_dtype=torch.float16
)

# Загружаем базовый пайплайн (можно указать путь к локальному .safetensors)
print("Загрузка базовой модели SD 1.5...")
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
    model_id, 
    motion_adapter=adapter, 
    torch_dtype=torch.float16
)

# Настройка планировщика
pipe.scheduler = DDIMScheduler.from_pretrained(
    model_id, subfolder="scheduler", clip_sample=False
)

# 3. ПОДКЛЮЧАЕМ IP-ADAPTER (Магия сохранения лица)
print("Подключение IP-Adapter Face...")
# Загружаем веса адаптера для удержания лица
pipe.load_ip_adapter(
    "h94/IP-Adapter", 
    subfolder="models", 
    weight_name="ip-adapter-full-face_sd15.bin"
)

# Настраиваем силу влияния IP-Adapter (Scale)
# Значение 0.7 - 0.8 жестко удерживает лицо, но оставляет ИИ свободу для анимации.
# Если лицо всё равно плывет — поднимите до 0.9. Если картинка ломается — снизьте до 0.6.
pipe.set_ip_adapter_scale(0.75)

# Оптимизации памяти для Tesla V100 / Mac
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# 4. Запускаем генерацию анимации
# Важно: в промпте мы больше НЕ описываем внешность человека (IP-Adapter уже знает её).
# Описываем только ДВИЖЕНИЕ и окружение.
prompt = "cinematic motion, hair blowing in the wind, wind rustling, bokeh, highly detailed, 4k"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, deformed face"

print("Оживление фотографии началось...")
output = pipe(
    video=init_frames,            # Наше псевдо-видео из фото
    ip_adapter_image=init_image,  # Передаем это же фото как визуальный промпт для лица!
    prompt=prompt,
    negative_prompt=negative_prompt,
    strength=0.65,                # Сила изменения кадров (чем меньше, тем ближе к оригиналу)
    guidance_scale=7.5,
    num_inference_steps=30,       # Чуть больше шагов для лучшей прорисовки деталей лица
    generator=torch.manual_seed(42) # Задаем seed для воспроизводимости
)

# 5. Сохраняем анимированное фото в GIF
output_path = "animated_face_fixed.gif"
export_to_gif(output.frames, output_path, fps=8)
print(f"Готово! Видео с сохраненным лицом сохранено в {output_path}")
