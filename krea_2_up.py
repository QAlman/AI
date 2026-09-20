import os
import torch
import gc
from diffusers import Krea2Pipeline
from pathlib import Path
from datetime import datetime
from PIL import Image

# 1. Системные настройки защиты памяти и офлайн-режима
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Пути
local_path = str(Path(r"D:\AI\IMG\krea2").as_posix())
output_dir = r"D:\AI\output_krea"
os.makedirs(output_dir, exist_ok=True)

# --- ГЛАВНЫЕ НАСТРОЙКИ (Промпт вынесен вверх для удобства) ---
prompt = "A high-tech cyberpunk room, chrome futuristic furniture, neon lighting reflected in dark glass, cinematic, professional photography, 8k resolution"
steps = 7
scale_factor = 4 # Во сколько раз увеличивать картинку (4х из 512 = 2048x2048)

print("1. Загрузка модели Krea 2 в память...")
gc.collect()
torch.cuda.empty_cache()

pipe = Krea2Pipeline.from_pretrained(
    local_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    ignore_mismatched_sizes=True
)

print("2. Включаем экстремальную послойную разгрузку памяти...")
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()
if hasattr(pipe, "vae") and pipe.vae is not None:
    pipe.vae.enable_slicing()

print("3. Запуск генерации Krea 2 Turbo...")
output = pipe(
    prompt=prompt, 
    num_inference_steps=steps, 
    guidance_scale=0.0, 
    height=512, 
    width=512
)

# Перестраховка на тип данных
final_image = output.images[0] if isinstance(output.images, list) else output.images

# 4. Создаем уникальное имя файлов по времени
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = f"krea2_{timestamp}.png"
upscale_name = f"krea2_{timestamp}_4K.png"
txt_name = f"krea2_{timestamp}_prompt.txt"

image_save_path = os.path.join(output_dir, base_name)
upscale_save_path = os.path.join(output_dir, upscale_name)
txt_save_path = os.path.join(output_dir, txt_name)

# Сохраняем базовый кадр
final_image.save(image_save_path)
print(f"🎉 Базовая картинка успешно сохранена: {base_name}")

# --- ФУНКЦИЯ СОХРАНЕНИЯ ПРОМПТА ---
print("📝 Запись промпта и настроек в текстовый файл...")
with open(txt_save_path, "w", encoding="utf-8") as f:
    f.write(f"--- ГЕНЕРАЦИЯ KREA 2 TURBO ---\n")
    f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"ПРОМПТ:\n{prompt}\n\n")
    f.write(f"НАСТРОЙКИ:\n")
    f.write(f"Шагов (num_inference_steps): {steps}\n")
    f.write(f"Базовый размер: 512x512\n")
    f.write(f"Размер после апскейла: {512*scale_factor}x{512*scale_factor}\n")
print(f"📝 Лог промпта сохранен: {txt_name}")

# --- МГНОВЕННЫЙ АПСКЕЙЛ LANCZOS ---
print(f"🤖 Запуск качественного масштабирования (Увеличение в {scale_factor}x)...")
new_width = final_image.width * scale_factor
new_height = final_image.height * scale_factor

try:
    upscaled_img = final_image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
except AttributeError:
    upscaled_img = final_image.resize((new_width, new_height), resample=Image.LANCZOS)

upscaled_img.save(upscale_save_path, quality=95)
print(f"🚀 Успешно! Четкая 4K картинка сохранена: {upscale_name}")
print("\nПолный цикл завершен! Все 3 файла лежат в папке.")
