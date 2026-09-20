import os
import torch
from PIL import Image
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def upscale_image(image_path, output_path, scale=4):
    """
    Функция увеличивает разрешение картинки в 4 раза (из 512x512 в 2048x2048 / 2K)
    Занимает всего 1-3 секунды на видеокарте 10 ГБ VRAM.
    """
    print(f"🤖 Запуск нейросетевого апскейла RealESRGAN (Увеличение в {scale}x)...")
    
    # 1. Загружаем картинку с диска
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # 2. Выбираем модель. Архитектура RRDBNet — стандарт для сверхчеткого апскейла
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # Репозиторий автоматически скачает маленькие веса модели (RealESRGAN_x4plus), если их нет
    upscaler = RealESRGANer(
        scale=4,
        model_path=None, # Библиотека сама подтянет нужный файл весов из сети в первый раз
        model=model,
        tile=0,          # 0 означает обработку картинки целиком (для 512х512 это мгновенно)
        tile_pad=10,
        pre_pad=0,
        half=True,       # Включаем FP16 (полуточность) для экономии VRAM и скорости
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 3. Запускаем апскейл (это займет пару секунд)
    output_np, _ = upscaler.enhance(img_np, outscale=scale)

    # 4. Сохраняем результат обратно в PIL-картинку
    final_image = Image.fromarray(output_np)
    final_image.save(output_path)
    print(f"🎉 Успешно! Четкая картинка сохранена в: {output_path}")

# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---
# Вы можете запустить этот блок для теста прямо сейчас:
if __name__ == "__main__":
    # Путь к вашей картинке 512х512
    input_file = r"D:\AI\output_krea\krea2_local_turbo_masterpiece.png" 
    # Путь, куда сохранить результат 2K/4K
    output_file = r"D:\AI\output_krea\krea2_local_masterpiece_4K.png"
    
    if os.path.exists(input_file):
        upscale_image(image_path=input_file, output_path=output_file, scale=4)
    else:
        print(f"Файл {input_file} не найден. Сначала сгенерируйте его через Krea 2 Turbo!")
