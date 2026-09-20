import os
from PIL import Image


def upscale_image_fast(image_path, output_path, scale=4):
    """
    Мгновенный и стабильный апскейл картинки с помощью высококачественного фильтра Lanczos.
    Работает за долю секунды, не выдает ошибок совместимости.
    """
    print(f"🤖 Запуск качественного масштабирования (Увеличение в {scale}x)...")

    # 1. Загружаем вашу картинку 512x512
    img = Image.open(image_path)

    # 2. Вычисляем новый размер (512 * 4 = 2048х2048 / 2K)
    new_width = img.width * scale
    new_height = img.height * scale

    # 3. Применяем фильтр Lanczos (высококачественное сглаживание пикселей)
    # В современных версиях PIL используется Image.Resampling.LANCZOS
    try:
        upscaled_img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
    except AttributeError:
        # Для совсем старых версий Pillow
        upscaled_img = img.resize((new_width, new_height), resample=Image.LANCZOS)

    # 4. Сохраняем результат в 4K
    upscaled_img.save(output_path, quality=95)  # Сохраняем с максимальным качеством JPG/PNG
    print(f"🎉 Успешно! Четкая картинка {new_width}x{new_height} сохранена в: {output_path}")


if __name__ == "__main__":
    # Укажите точный путь к вашей сгенерированной картинке от Krea 2 Turbo
    input_file = r"D:\AI\IMG\base_image_2.png"
    output_file = r"D:\AI\IMG\base_image_3.png"

    if os.path.exists(input_file):
        upscale_image_fast(input_file, output_file, scale=4)
    else:
        print(f"Файл {input_file} не найден. Проверьте правильность пути!")
