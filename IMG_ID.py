import torch
from PIL import Image
from diffusers import StableDiffusionXLInstantIDPipeline, ControlNetModel
from diffusers.utils import load_image

# Инициализация путей к файлам (замените на свои)
BASE_MODEL_PATH = r"C:\AI\models\juggernautXL_v9.safetensors"
CONTROLNET_PATH = r"C:\AI\models\controlnet_instantid_sdxl.safetensors"
IP_ADAPTER_PATH = r"C:\AI\models\ip-adapter-instantid-sdxl.bin"

INPUT_IMAGE_PATH = "my_face_photo.jpg"
OUTPUT_IMAGE_PATH = "result_high_strength.jpg"

def generate_with_instantid():
    print("⏳ Загрузка компонентов InstantID и базовой модели...")
    
    # 1. Загружаем специализированный ControlNet для лица
    # Стало для вашего нового файла
    controlnet = ControlNetModel.from_single_file(
        CONTROLNET_PATH, 
        torch_dtype=torch.float16,
        use_safetensors=True  # Обязательно добавьте этот флаг
    )

    
    # 2. Загружаем основной пайплайн на базе вашей модели с Civitai
    pipe = StableDiffusionXLInstantIDPipeline.from_single_file(
        BASE_MODEL_PATH,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # 3. Подгружаем веса IP-Adapter для InstantID
    # Передаем путь к папке и имя файла отдельно
    pipe.load_ip_adapter_instantid(IP_ADAPTER_PATH)
    pipe.to("cuda")
    pipe.enable_model_cpu_offload() # Экономия VRAM

    print("📸 Обработка оригинального фото...")
    # Загружаем изображение лица (оно будет и источником личности, и основой для композиции)
    init_image = Image.open(INPUT_IMAGE_PATH).convert("RGB").resize((1024, 1024))
    
    # В InstantID в качестве референса лица и карты структуры используется изображение.
    # Мы передаем одно и то же фото, чтобы сохранить позу и лицо, но полностью заменить окружение.
    face_image = init_image 
    
    # Настройки промпта для полной замены одежды и фона
    # Описываем финальную картинку с нуля
    prompt = "A cinematic medium shot of a man wearing a futuristic cyber-punk jacket, standing neon-lit rainy street of Tokyo at night, photorealistic, 8k resolution, dramatic look"
    negative_prompt = "bad anatomy, deformed, canvas, art, drawing, low quality, blurry"
    
    print("🚀 Запуск генерации с жесткой фиксацией лица...")
    
    # Запуск
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_image,       # Отвечает за копирование черт лица
        image=face_image,              # Отвечает за удержание позы / структуры
        controlnet_conditioning_scale=0.8, # Сила удержания геометрии лица (0.5 - 0.9)
        num_inference_steps=30,
        guidance_scale=7.0,
    ).images

    images[0].save(OUTPUT_IMAGE_PATH)
    print(f"✅ Успешно сохранено в {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        generate_with_instantid()
    else:
        print("❌ Требуется видеокарта с поддержкой CUDA!")
