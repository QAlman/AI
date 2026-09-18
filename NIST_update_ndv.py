import os
import json
import time
import requests
import urllib3
from datetime import datetime, timedelta, timezone

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = "0bdbbd2b-9cca-47a9-b128-5677709f01de"
BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
UPDATES_DIR = "nvd_updates"
os.makedirs(UPDATES_DIR, exist_ok=True)

# Задаем окно за последние 25 часов (с запасом в 1 час на синхронизацию)
now = datetime.now(timezone.utc)
yesterday = now - timedelta(hours=25)

# Формат времени для NVD API строго: YYYY-MM-DDTHH:MM:SS.SSS
last_mod_start = yesterday.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
last_mod_end = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

params = {
    "lastModStartDate": last_mod_start,
    "lastModEndDate": last_mod_end,
    "resultsPerPage": 2000,
    "startIndex": 0
}

headers = {"apiKey": API_KEY, "User-Agent": "NVD-Updater"}

print(f"Запрос обновлений с {last_mod_start} по {last_mod_end}...")

try:
    response = requests.get(BASE_URL, headers=headers, params=params, verify=False, timeout=30)
    if response.status_code == 200:
        data = response.json()
        vulnerabilities = data.get("vulnerabilities", [])

        if vulnerabilities:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(UPDATES_DIR, f"update_{timestamp}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(vulnerabilities, f, ensure_ascii=False, indent=2)
            print(f"Успешно сохранено {len(vulnerabilities)} измененных/новых CVE в {filename}")
            # ТУТ ВАШ КОД: отправить эти CVE на обновление в вашу базу данных
        else:
            print("Новых обновлений за этот период нет.")
    else:
        print(f"Ошибка API: {response.status_code}")
except Exception as e:
    print(f"Ошибка при обновлении: {e}")
