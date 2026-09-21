import os
import torch
import gc
from diffusers import WanPipeline
from pathlib import Path
from datetime import datetime
import imageio
import numpy as np

# 1. Системные настройки защиты памяти и офлайн-режима
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Настройка путей
local_path = str(Path(r"D:\AI\IMG\wan2").as_posix())
output_dir = r"D:\AI\output_video"
os.makedirs(output_dir, exist_ok=True)

# Функция безопасного чтения промпта из файла (наша доработанная версия)
def read_prompt_file(file_path, default_text=""):
    if not os.path.exists(file_path):
        return default_text
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            clean_lines.append(" ".join(stripped.split()))
    return ", ".join(clean_lines)

# Читаем промпт из корня (если файла нет, сработает дефолтный)
prompt = read_prompt_file("my_video_prompt.txt", default_text="A cinematic shot of a futuristic sports car driving through a neon cyberpunk city at night, rain reflections, 4k resolution")

print("1. Загрузка видео-модели Wan2.1 в память...")
gc.collect()
torch.cuda.empty_cache()

# Загружаем пайплайн Wan Video
pipe = WanPipeline.from_pretrained(
    local_path,
    torch_dtype=torch.bfloat16, # Базовая точность вычислений
    local_files_only=True
)

print("2. Включаем МАКСИМАЛЬНУЮ оптимизацию памяти под 10 ГБ VRAM...")
# Послойная разгрузка — наш главный щит от ошибок OOM
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()

print("3. Запуск генерации видео-кадров...")
# Генерируем небольшое тестовое разрешение для экономии VRAM
# Wan2.1 T2V нативно генерирует потрясающие кадры на этих настройках
output = pipe(
    prompt=prompt,
    num_frames=16,             # 16 кадров — отличная стартовая длина для теста
    height=480,                # Безопасное экономное разрешение для 10 ГБ VRAM
    width=832,                 # Кинематографичный широкий формат
    num_inference_steps=30,    # Для Wan обычно нужно 30-40 шагов
    guidance_scale=5.0
)
video_frames = output.frames

# 4. Создаем уникальные и идентичные имена файлов по времени
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = f"wan_{timestamp}.mp4"
txt_name = f"wan_{timestamp}.txt"

video_save_path = os.path.join(output_dir, video_name)
txt_save_path = os.path.join(output_dir, txt_name)

# 5. Сохраняем текстовый лог промпта и модели
print("📝 Запись промпта в текстовый файл...")
model_name_only = os.path.basename(local_path)
with open(txt_save_path, "w", encoding="utf-8") as f:
    f.write(f"--- ГЕНЕРАЦИЯ ВИДЕО WAN2.1 ---\n")
    f.write(f"Файл видео: {video_name}\n")
    f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Используемая модель: {model_name_only}\n\n")
    f.write(f"ПРОМПТ:\n{prompt}\n\n")
    f.write(f"НАСТРОЙКИ:\n")
    f.write(f"Кадров (num_frames): 16\n")
    f.write(f"Разрешение: 832x480\n")

# 6. Конвертируем кадры в NumPy и собираем плавное видео
print("🎬 Сборка видеофайла MP4...")
np_frames = [np.array(f.convert("RGB")) for f in video_frames]
# Ставим fps=8 или 10, чтобы 16 кадров шли плавно и не пролетали за доли секунды
imageio.mimsave(video_save_path, np_frames, fps=10, format="FFMPEG")

print(f"\n🎉 ПРАЗДНИК! Видео успешно создано локально и сохранено в: {video_save_path}")
