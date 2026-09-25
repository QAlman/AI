import os
import re
import json
import subprocess
import tempfile
from pathlib import Path


def get_current_system_policies():
    """
    Экспортирует текущую локальную политику безопасности Windows во временный файл
    и возвращает его содержимое в виде строки.
    """
    print("[INFO] Сбор текущих политик безопасности системы через secedit...")
    temp_dir = tempfile.gettempdir()
    temp_inf = os.path.join(temp_dir, "current_system_policy.inf")

    # Экспортируем сразу все области (и права пользователей, и политики паролей)
    secedit_cmd = f'secedit /export /cfg "{temp_inf}"'

    try:
        result = subprocess.run(secedit_cmd, shell=True, capture_output=True, text=True, encoding='cp866')

        if result.returncode != 0:
            print(f"[ERROR] Не удалось запустить secedit. Код: {result.returncode}")
            print(f"[ERROR] Подсказка: Убедитесь, что скрипт запущен ОТ ИМЕНИ АДМИНИСТРАТОРА.")
            return None

        if not os.path.exists(temp_inf):
            return None

        # Читаем файл с учетом возможной кодировки UTF-16LE, используемой Windows
        try:
            with open(temp_inf, 'r', encoding='utf-16') as f:
                content = f.read()
        except UnicodeError:
            with open(temp_inf, 'r', encoding='cp866') as f:
                content = f.read()
        return content

    finally:
        # Корректно заметаем следы
        if os.path.exists(temp_inf):
            try:
                os.remove(temp_inf)
            except Exception:
                pass


def check_policy_value(tech_key, file_content):
    """
    Ищет значение указанного tech_key внутри файла конфигурации secedit
    """
    if not tech_key:
        return None
    # Ищем строчку вида: TechKey = Значение
    pattern = rf"^{tech_key}\s*=\s*(.*)$"
    match = re.search(pattern, file_content, re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def run_audit(json_path):
    # 1. Загружаем базу знаний CIS из нашего JSON
    if not os.path.exists(json_path):
        print(f"[ERROR] База знаний не найдена по пути: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)

    # 2. Собираем снимок системы
    policy_content = get_current_system_policies()
    if not policy_content:
        print("[ERROR] Не удалось собрать данные системы. Выход.")
        return

    print(f"[INFO] Загружено правил для проверки: {len(rules)}")
    print("-" * 80)
    print(f"{'ID':<10} | {'Тип проверки':<15} | {'Статус':<10} | {'Описание правила'}")
    print("-" * 80)

    stats = {"PASS": 0, "FAIL": 0, "UNKNOWN": 0}

    # 3. Основной цикл аудита
    for rule in rules:
        rule_id = rule["id"]
        rule_type = rule["type"]
        tech_key = rule["tech_key"]
        expected = rule["expected_value"]
        title = rule["title"]

        # Обрезаем слишком длинный тайтл для красивого вывода в консоль
        short_title = title[:50] + "..." if len(title) > 50 else title

        # Логика для политик паролей и прав пользователей (работают через secedit)
        if rule_type in ["user_right", "account_policy"]:
            actual_value = check_policy_value(tech_key, policy_content)

            if actual_value is None:
                # Если ключа нет в файле, для прав пользователей это обычно значит "Никто"
                if rule_type == "user_right":
                    status = "FAIL" if expected else "PASS"
                else:
                    status = "UNKNOWN"
            else:
                # Проверка по типу соответствия
                if rule["match_type"] == "contains":
                    # Для прав пользователей проверяем наличие SID или имени (например, Guests)
                    if expected.lower() in actual_value.lower() or "guests" in actual_value.lower():
                        status = "PASS"
                    else:
                        status = "FAIL"
                else:
                    # Для точного совпадения (например, длина пароля должна быть 24)
                    if expected and actual_value == expected:
                        status = "PASS"
                    else:
                        status = "FAIL"

        else:
            # Сюда будут попадать реестр и другие типы, которые мы еще не реализовали
            status = "UNKNOWN"

        stats[status] += 1
        print(f"{rule_id:<10} | {rule_type:<15} | {status:<10} | {short_title}")

    print("-" * 80)
    print(
        f"[РЕЗУЛЬТАТ] Аудит завершен. Успешно (PASS): {stats['PASS']} | Провалено (FAIL): {stats['FAIL']} | Не определено (UNKNOWN): {stats['UNKNOWN']}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    # Путь к нашему сгенерированному JSON в папке DATA
    json_db = script_dir.parent / "DATA" / "cis_benchmarks.json"

    run_audit(json_db)
