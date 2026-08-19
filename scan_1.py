import os
import json
import csv

# Путь к распакованному репозиторию cvelistV5
json_dir = "./cvelistV5-main" 
output_csv = "C:/Program Files/Nmap/scripts/vulscan/cve.csv"

print("Конвертация JSON в CSV для vulscan...")
with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';') # vulscan обычно использует ';' или ','
    
    # Рекурсивно обходим все папки с JSON
    for root, dirs, files in os.walk(json_dir):
        for file in files:
            if file.endswith('.json') and file.startswith('CVE-'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as jf:
                    try:
                        data = json.load(jf)
                        cve_id = data['cveMetadata']['cveId']
                        
                        # Достаем описание (английское)
                        descriptions = data['containers']['cna']['descriptions']
                        desc_text = next((d['value'] for d in descriptions if d['lang'] == 'en'), "")
                        
                        # vulscan требует формат: ИДЕНТИФИКАТОР;ОПИСАНИЕ
                        writer.writerow([cve_id, desc_text.replace('\n', ' ')])
                    except Exception:
                        continue
print("Готово! База данных cve.csv обновлена.")
