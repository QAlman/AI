import os
import json
import sqlite3
from pathlib import Path


# --- 1. Инициализация Базы Данных ---
def init_db(db_path="nvd_complete.db"):
    """Создает правильную схему БД со связями и индексами"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Таблица для самих CVE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cves (
            cve_id TEXT PRIMARY KEY,
            description TEXT
        )
    """)

    # Таблица для CPE (продукты, точные версии и диапазоны)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cpe_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cve_id TEXT,
            vendor TEXT,
            product TEXT,
            exact_version TEXT,
            v_start_inc TEXT,
            v_start_exc TEXT,
            v_end_inc TEXT,
            v_end_exc TEXT,
            FOREIGN KEY (cve_id) REFERENCES cves(cve_id)
        )
    """)

    # Индексы для мгновенного поиска по вендору/продукту
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vendor_prod ON cpe_matches(vendor, product);")
    conn.commit()
    return conn


# --- 2. Вспомогательные функции ---
def parse_cpe_string(cpe_criteria):
    """Разбивает CPE 2.3 строку на компоненты"""
    parts = cpe_criteria.split(":")
    if len(parts) >= 6:
        return parts[3], parts[4], parts[5]  # vendor, product, version
    return None, None, None


# --- 3. Главный парсер папок и файлов ---
def build_db_from_folders(root_folder_path, db_path="nvd_complete.db"):
    """Рекурсивно обходит папки, читает все JSON и пишет в SQLite"""
    conn = init_db(db_path)
    cursor = conn.cursor()

    # Считаем счетчики для вывода прогресса
    total_files_processed = 0
    total_cves_saved = 0

    # Рекурсивный поиск всех .json файлов во всех подпапках
    root_path = Path(root_folder_path)
    json_files = list(root_path.glob("**/*.json"))

    if not json_files:
        print(f"⚠️ В папке '{root_folder_path}' не найдено файлов .json")
        return

    print(f"🗂️ Найдено JSON-файлов для обработки: {len(json_files)}")
    print("🚀 Начинаем импорт...")

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Ошибка чтения файла {file_path}: {e}")
            continue

        # --- НАЧАЛО ИСПРАВЛЕНИЯ ---
        # Проверяем, что именно прилетело из JSON
        if isinstance(data, dict):
            # Если это словарь (как в API), ищем ключ "vulnerabilities"
            vulnerabilities = data.get("vulnerabilities", [])
        elif isinstance(data, list):
            # Если это готовый список CVE, берем его целиком
            vulnerabilities = data
        else:
            # Если структура неизвестна, пропускаем файл
            continue

        if not vulnerabilities:
            continue
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---


        for item in vulnerabilities:
            cve_data = item.get("cve", {})
            cve_id = cve_data.get("id")
            if not cve_id:
                continue

            # Извлекаем английское описание
            descriptions = cve_data.get("descriptions", [])
            description_text = next((d["value"] for d in descriptions if d["lang"] == "en"), "")

            # Сохраняем/обновляем CVE
            cursor.execute(
                "INSERT OR IGNORE INTO cves (cve_id, description) VALUES (?, ?)",
                (cve_id, description_text)
            )
            total_cves_saved += 1

            # Разбираем конфигурации уязвимого ПО
            configurations = cve_data.get("configurations", [])
            for config in configurations:
                for node in config.get("nodes", []):
                    for match in node.get("cpeMatch", []):
                        cpe_criteria = match.get("criteria", "")
                        vendor, product, exact_version = parse_cpe_string(cpe_criteria)

                        if not vendor or not product:
                            continue

                        # Если указана конкретная уязвимая версия ПО
                        if exact_version not in ["*", "-"]:
                            cursor.execute("""
                                INSERT INTO cpe_matches (cve_id, vendor, product, exact_version)
                                VALUES (?, ?, ?, ?)
                            """, (cve_id, vendor, product, exact_version))
                            continue

                        # Если указаны плавающие диапазоны версий (до/после какой-то версии)
                        v_start_inc = match.get("versionStartIncluding")
                        v_start_exc = match.get("versionStartExcluding")
                        v_end_inc = match.get("versionEndIncluding")
                        v_end_exc = match.get("versionEndExcluding")

                        if any([v_start_inc, v_start_exc, v_end_inc, v_end_exc]):
                            cursor.execute("""
                                INSERT INTO cpe_matches (
                                    cve_id, vendor, product, 
                                    v_start_inc, v_start_exc, v_end_inc, v_end_exc
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (cve_id, vendor, product, v_start_inc, v_start_exc, v_end_inc, v_end_exc))

        total_files_processed += 1
        # Фиксируем изменения в базе каждые 50 файлов для экономии ресурсов
        if total_files_processed % 50 == 0:
            conn.commit()
            print(f"📈 Обработано файлов: {total_files_processed}/{len(json_files)}...")

    # Финальный коммит и закрытие
    conn.commit()
    conn.close()
    print("\n✅ Импорт успешно завершен!")
    print(f"📊 Всего обработано файлов: {total_files_processed}")
    print(f"💾 Создана база данных с правильной структурой.")
# 1. Укажите путь к корневой папке, где лежат ваши папки/подпапки с JSON-файлами API
PATH_TO_JSON_DATA = "./nvd_database"

# 2. Собираем базу (запускается один раз)
build_db_from_folders(PATH_TO_JSON_DATA, db_path="my_smart_nvd.db")
