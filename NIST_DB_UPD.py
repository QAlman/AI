import json
import sqlite3
from pathlib import Path


# --- 1. Вспомогательная функция парсинга CPE ---
def parse_cpe_string(cpe_criteria):
    """Разбивает CPE 2.3 строку на компоненты"""
    parts = cpe_criteria.split(":")
    if len(parts) >= 6:
        return parts[3], parts[4], parts[5]  # vendor, product, version
    return None, None, None


# --- 2. Быстрая функция обновления базы ---
def update_db_from_changes_fast(update_folder_path, db_path="my_smart_nvd_old.db"):
    """Супер-быстрое обновление БД с использованием PRAGMA, индексов и склейкой всех CWE"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- НАСТРОЙКИ УСКОРЕНИЯ (PRAGMA) ---
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")
    cursor.execute("PRAGMA cache_size = -64000;")

    # --- АВТОМАТИЧЕСКАЯ ПРОВЕРКА И НАСТРОЙКА СТРУКТУРЫ БД ---
    # Создаем критический индекс для быстрой очистки CPE связей
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cpe_matches_cve_id ON cpe_matches(cve_id);")

    # Проверяем, существует ли уже колонка cwe_id, чтобы скрипт не падал
    cursor.execute("PRAGMA table_info(cves);")
    columns = [col[1] for col in cursor.fetchall()]
    if "cwe_id" not in columns:
        print("🔧 Добавляю отсутствующую колонку cwe_id в таблицу cves...")
        cursor.execute("ALTER TABLE cves ADD COLUMN cwe_id TEXT;")

    # Создаем индекс для мгновенного поиска по CWE
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cves_cwe_id ON cves(cwe_id);")

    conn.commit()

    root_path = Path(update_folder_path)
    json_files = list(root_path.glob("**/*.json"))

    if not json_files:
        print(f"⚠️ В папке обновлений '{update_folder_path}' не найдено файлов .json")
        conn.close()
        return

    print(f"🔄 Найдено файлов с обновлениями: {len(json_files)}")
    print("🚀 Начинаем оптимизированный импорт...")

    total_files = 0
    total_updates = 0

    cves_to_insert = []
    cpes_to_insert = []

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Ошибка чтения файла {file_path}: {e}")
            continue

        if isinstance(data, dict):
            vulnerabilities = data.get("vulnerabilities", [])
        elif isinstance(data, list):
            vulnerabilities = data
        else:
            continue

        for item in vulnerabilities:
            cve_data = item.get("cve", {})
            cve_id = cve_data.get("id")
            if not cve_id:
                continue

            descriptions = cve_data.get("descriptions", [])
            description_text = next((d["value"] for d in descriptions if d["lang"] == "en"), "")

            # --- УМНЫЙ СБОР И СКЛЕЙКА ВСЕХ CWE ---
            weaknesses = cve_data.get("weaknesses", [])
            found_cwes = []

            # 1. Вытягиваем все значения CWE из JSON структуры
            for w in weaknesses:
                descriptions_cwe = w.get("description", [])
                for d in descriptions_cwe:
                    val = d.get("value")
                    if val and val not in found_cwes:  # Избегаем дубликатов в рамках одной уязвимости
                        found_cwes.append(val)

            # 2. Фильтруем мусор
            # Оставляем только реальные коды CWE, если они есть
            real_cwes = [c for c in found_cwes if c.startswith("CWE-")]

            if real_cwes:
                # Склеиваем реальные CWE через запятую
                cwe_id_str = ", ".join(real_cwes)
            elif found_cwes:
                # Если реальных кодов нет вообще, сохраняем то, что пришло (например, NVD-CWE-Other)
                cwe_id_str = ", ".join(found_cwes)
            else:
                cwe_id_str = None

            # 1. Готовим данные для CVE (добавляем в буфер)
            cves_to_insert.append((cve_id, description_text, cwe_id_str))

            # 2. Мгновенно удаляем старые связи
            cursor.execute("DELETE FROM cpe_matches WHERE cve_id = ?", (cve_id,))

            # 3. Разбираем конфигурации продуктов
            configurations = cve_data.get("configurations", [])
            for config in configurations:
                for node in config.get("nodes", []):
                    for match in node.get("cpeMatch", []):
                        cpe_criteria = match.get("criteria", "")
                        vendor, product, exact_version = parse_cpe_string(cpe_criteria)

                        if not vendor or not product:
                            continue

                        if exact_version not in ["*", "-"]:
                            cpes_to_insert.append((cve_id, vendor, product, exact_version, None, None, None, None))
                            continue

                        v_start_inc = match.get("versionStartIncluding")
                        v_start_exc = match.get("versionStartExcluding")
                        v_end_inc = match.get("versionEndIncluding")
                        v_end_exc = match.get("versionEndExcluding")

                        if any([v_start_inc, v_start_exc, v_end_inc, v_end_exc]):
                            cpes_to_insert.append(
                                (cve_id, vendor, product, None, v_start_inc, v_start_exc, v_end_inc, v_end_exc))

            total_updates += 1

        total_files += 1

        # ПАКЕТНАЯ ЗАПИСЬ: Сбрасываем накопленные данные в базу каждые 50 файлов
        if total_files % 50 == 0:
            if cves_to_insert:
                cursor.executemany(
                    "INSERT OR REPLACE INTO cves (cve_id, description, cwe_id) VALUES (?, ?, ?)",
                    cves_to_insert
                )
                cves_to_insert.clear()

            if cpes_to_insert:
                cursor.executemany("""
                    INSERT INTO cpe_matches (
                        cve_id, vendor, product, exact_version, 
                        v_start_inc, v_start_exc, v_end_inc, v_end_exc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, cpes_to_insert)
                cpes_to_insert.clear()

            conn.commit()
            print(f"📈 Обработано файлов: {total_files}/{len(json_files)}...")

    # Финальный сброс остатков данных из буферов
    if cves_to_insert:
        cursor.executemany(
            "INSERT OR REPLACE INTO cves (cve_id, description, cwe_id) VALUES (?, ?, ?)",
            cves_to_insert
        )
    if cpes_to_insert:
        cursor.executemany("""
            INSERT INTO cpe_matches (
                cve_id, vendor, product, exact_version, 
                v_start_inc, v_start_exc, v_end_inc, v_end_exc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, cpes_to_insert)

    conn.commit()
    conn.close()

    print(f"\n✅ База данных успешно обновлена за пару мгновений!")
    print(f"📊 Всего обработано файлов обновлений: {total_files}")
    print(f"🔄 Добавлено/обновлено записей CVE: {total_updates}")


# --- 3. Точка запуска ---
if __name__ == "__main__":
    PATH_TO_UPDATES = "./nvd_updates"
    update_db_from_changes_fast(PATH_TO_UPDATES, db_path="my_smart_nvd.db")
