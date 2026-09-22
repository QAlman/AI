import torch
from PIL import Image
from diffusers import AnimateDiffSDXLPipeline, MotionAdapter
from diffusers.utils import export_to_gif
from rife_ncnn_vulkan_python import Rife

# --- НАСТРОЙКИ ПУТЕЙ ---
PATH_TO_MOTION_ADAPTER = "./models/animatediff_motion_sdxl_beta.safetensors"
PATH_TO_SDXL_CHECKPOINT = "./models/pony_diffusion_v6_xl.safetensors" 
image_path = "./my_photo.jpg"

# 1. ПОДГОТОВКА КАДРОВ
init_image = Image.open(image_path).convert("RGB").resize((1024, 1024))
init_frames = [init_image] * 16

# 2. СБОРКА ПАЙПЛАЙНА АНИМАЦИИ
print("Загрузка моделей...")
adapter = MotionAdapter.from_single_file(PATH_TO_MOTION_ADAPTER, torch_dtype=torch.float16)
pipe = AnimateDiffSDXLPipeline.from_single_file(
    PATH_TO_SDXL_CHECKPOINT, motion_adapter=adapter, torch_dtype=torch.float16, use_safetensors=True
)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.bin")
pipe.set_ip_adapter_scale(0.75)

# Оптимизация памяти
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# 3. ГЕНЕРАЦИЯ БАЗОВЫХ 16 КАДРОВ
print("Шаг 1: Генерация базовой анимации (16 кадров)...")
prompt = "cinematic motion, hair blowing in the wind, soft studio lighting, highly detailed, 4k"
negative_prompt = "bad quality, low quality, deformed face, blurry, static"

output = pipe(
    video=init_frames,
    ip_adapter_image=init_image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    strength=0.65,
    guidance_scale=6.5,
    num_inference_steps=25,
    generator=torch.manual_seed(42)
)
base_frames = output.frames  # Здесь у нас ровно 16 статичных кадров

# 4. ИНТЕРПОЛЯЦИЯ (СГЛАЖИВАНИЕ ЧЕРЕЗ RIFE)
print("Шаг 2: Запуск сглаживания кадров через RIFE...")
# Инициализируем RIFE (он автоматически подтянет нужную модель)
rife = Rife(gpuid=0, fast_mode=True)

smoothed_frames = []
# Проходимся по парам кадров и генерируем между ними по 3 дополнительных кадра
for i in range(len(base_frames) - 1):
    frame1 = base_frames[i]
    frame2 = base_frames[i + 1]
    
    # Делаем апскейл плавности х4 (между каждыми двумя кадрами вставится еще три)
    interpolated = rife.process(frame1, frame2, times=4)
    
    # Добавляем первый кадр и сгенерированные промежуточные
    smoothed_frames.append(frame1)
    smoothed_frames.extend(interpolated)

# Добавляем самый последний кадр в конец видео
smoothed_frames.append(base_frames[-1])

# 5. СОХРАНЕНИЕ РЕЗУЛЬТАТА
output_path = "smooth_animation_24fps.gif"
# Выставляем fps=24, так как кадров стало в 4 раза больше (всего 61 кадр)
export_to_gif(smoothed_frames, output_path, fps=24)
print(f"Успешно! Плавная анимация сохранена в файл: {output_path} (Всего кадров: {len(smoothed_frames)})")
