import sqlite3
import socket
from urllib.parse import urlparse
import nmap

# Название вашей созданной базы данных
DB_NAME = "my_smart_nvd.db"


# =====================================================================
# 1. СЛУЖЕБНЫЕ ФУНКЦИИ И СЕМАНТИЧЕСКОЕ СРАВНЕНИЕ ВЕРСИЙ
# =====================================================================

def compare_versions(v1, v2):
    """Семантическое сравнение версий для SQLite (возвращает -1, 0, 1)"""
    if v1 is None or v2 is None:
        return 0

    def to_list(v):
        return [int(x) if x.isdigit() else x for x in v.replace('-', '.').split('.')]

    try:
        parts1, parts2 = to_list(v1), to_list(v2)
    except ValueError:
        return -1 if v1 < v2 else (1 if v1 > v2 else 0)

    len_diff = len(parts1) - len(parts2)
    if len_diff > 0:
        parts2.extend([0] * len_diff)
    elif len_diff < 0:
        parts1.extend([0] * abs(len_diff))

    for p1, p2 in zip(parts1, parts2):
        if isinstance(p1, int) and isinstance(p2, int):
            if p1 < p2: return -1
            if p1 > p2: return 1
        else:
            if str(p1) < str(p2): return -1
            if str(p1) > str(p2): return 1
    return 0


def extract_host(target_input):
    """Очищает URL до чистого домена/IP и определяет итоговый хост для Nmap"""
    target_clean = target_input.strip()

    # Если передали URL (например, https://example.com), извлекаем только домен
    if "://" in target_clean or target_clean.startswith("www."):
        if "://" not in target_clean:
            target_clean = "http://" + target_clean
        parsed_url = urlparse(target_clean)
        # Извлекаем хост и отбрасываем порт, если он указан в URL (например, :8080)
        target_clean = parsed_url.netloc.split(":")[0]

    # Разрешаем имя домена в IP-адрес для корректной проверки nm.all_hosts()
    try:
        ip_address = socket.gethostbyname(target_clean)
        print(f"[+] Цель успешно определена: {target_clean} -> IP: {ip_address}")
        return ip_address
    except socket.gaierror:
        # Если имя локальное и не резолвится через внешние DNS, возвращаем как есть
        return target_clean


# =====================================================================
# 2. ПОИСК УЯЗВИМОСТЕЙ В СВЯЗАННОЙ БАЗЕ ДАННЫХ SQLite
# =====================================================================

def check_cve_for_product(product_name, version=None):
    """Поиск уязвимостей в локальной базе с нормализацией имен Nmap -> CPE"""
    conn = sqlite3.connect(DB_NAME)
    conn.create_function("COMPARE", 2, compare_versions)
    cursor = conn.cursor()

    prod_lower = product_name.lower().strip()

    nmap_to_cpe_map = {
        "microsoft iis httpd": "internet_information_services",
        "apache httpd": "http_server",
        "nginx": "nginx",
        "openssh": "openssh"
    }

    if prod_lower in nmap_to_cpe_map:
        product_clean = nmap_to_cpe_map[prod_lower]
    else:
        product_clean = prod_lower.replace(" ", "_")

    if version:
        version_clean = version.lower().strip()
        # ДОБАВЛЕНО ПОЛЕ c.cwe_id В SELECT
        query = """
        SELECT DISTINCT c.cve_id, c.description, c.cwe_id
        FROM cves c
        JOIN cpe_matches m ON c.cve_id = m.cve_id
        WHERE m.product LIKE ? AND (
            m.exact_version = ? 
            OR (
                (m.v_start_inc IS NULL OR COMPARE(?, m.v_start_inc) >= 0) AND
                (m.v_start_exc IS NULL OR COMPARE(?, m.v_start_exc) > 0) AND
                (m.v_end_inc IS NULL   OR COMPARE(?, m.v_end_inc) <= 0) AND
                (m.v_end_exc IS NULL   OR COMPARE(?, m.v_end_exc) < 0) AND
                (m.v_start_inc IS NOT NULL OR m.v_start_exc IS NOT NULL OR m.v_end_inc IS NOT NULL OR m.v_end_exc IS NOT NULL)
            )
        )
        LIMIT 5
        """
        cursor.execute(query, (f"%{product_clean}%", version_clean, version_clean, version_clean, version_clean,
                               version_clean))
    else:
        # ДОБАВЛЕНО ПОЛЕ c.cwe_id В SELECT
        query = """
        SELECT DISTINCT c.cve_id, c.description, c.cwe_id
        FROM cves c
        JOIN cpe_matches m ON c.cve_id = m.cve_id
        WHERE m.product LIKE ?
        LIMIT 5
        """
        cursor.execute(query, (f"%{product_clean}%",))

    rows = cursor.fetchall()
    conn.close()
    return rows # Теперь возвращает кортежи вида (cve_id, description, cwe_id)



# =====================================================================
# 3. ОСНОВНОЙ ЦИКЛ СКАНИРОВАНИЯ И АНАЛИЗА ХОСТА
# =====================================================================

def run_scan_and_analyze():
    # ТЕСТОВАЯ ЦЕЛЬ: Сюда можно вставить IP, чистый домен или полную ссылку URL
    target_input = "https://.ru/"

    # Имя файла, куда запишется текстовый отчет
    report_filename = "nmap_cwe_report.txt"

    print(f"=== Инициализация Nmap ===")
    target = extract_host(target_input)

    nm = nmap.PortScanner()
    print(f"Запуск сканирования хоста {target} через Nmap... (это может занять время)")

    try:
        nm.scan(hosts=target, arguments='-sV -F -Pn -v --version-intensity 7')
    except nmap.PortScannerError as e:
        print(f"❌ Ошибка Nmap: {e}")
        return
    except Exception as e:
        print(f"❌ Непредвиденная ошибка при работе Nmap: {e}")
        return

    print(f"\n[+] Сканирование завершено. Идет генерация отчета в файл: {report_filename}")

    # Открываем файл для записи с кодировкой UTF-8, чтобы не было проблем с русскими буквами
    with open(report_filename, "w", encoding="utf-8") as f:

        # Пишем заголовок отчета
        f.write("=========================================================\n")
        f.write("      ОТЧЕТ ОБ УЯЗВИМОСТЯХ БЕЗОПАСНОСТИ СЕТИ (Nmap + CWE)\n")
        f.write("=========================================================\n\n")
        f.write(f"Цель сканирования: {target_input} ({target})\n")
        f.write("Статус хоста: АКТИВЕН\n\n")
        f.write("=== АНАЛИЗ РЕЗУЛЬТАТОВ СКАНЕРОМ ===\n")

        if target not in nm.all_hosts():
            f.write(f"❌ Хост {target} не отвечает на запросы сканера или все порты закрыты.\n")
            return

        for proto in nm[target].all_protocols():
            lport = nm[target][proto].keys()

            for port in sorted(lport):
                port_data = nm[target][proto][port]
                state = port_data.get('state')

                # Анализируем исключительно открытые сетевые порты
                if state == 'open':
                    product = port_data.get('product', '')
                    version = port_data.get('version', '')
                    extrainfo = port_data.get('extrainfo', '')

                    f.write(f"\n[+] Порт {port}/{proto} ОТКРЫТ\n")
                    f.write(f"    Сервис: {port_data.get('name', 'unknown')}\n")

                    if product:
                        f.write(
                            f"    Обнаружено ПО: {product} (Версия: {version if version else 'не определена'}) {extrainfo}\n")
                        f.write("    Поиск уязвимостей в локальной базе NVD SQLite...\n")

                        # Вызов функции умного поиска CVE по нашей схеме данных
                        vulnerabilities = check_cve_for_product(product, version)

                        if vulnerabilities:
                            f.write(f"    ⚠️ Найдено уязвимостей (показаны первые 5):\n")
                            for cve in vulnerabilities:
                                cve_id = cve[0]
                                description = cve[1]
                                # Безопасно извлекаем CWE, если он есть в вашей базе
                                cwe_id = cve[2] if len(cve) > 2 and cve[2] else "Не указан"

                                f.write(f"    [!] Уязвимость: {cve_id} -> [Тип ошибки: {cwe_id}]\n")
                                f.write(f"        Описание: {description[:140]}...\n")

                                # Добавляем в файл ссылку на MITRE для удобства руководства
                                if cwe_id and "CWE-" in str(cwe_id):
                                    cwe_num = ''.join(filter(str.isdigit, str(cwe_id)))
                                    f.write(f"        Подробнее о типе ошибки: https://mitre.org{cwe_num}.html\n")
                        else:
                            f.write("    ✅ Известных CVE для этого продукта/версии в базе не найдено.\n")
                    else:
                        f.write("    ⚠ ПО и версия не распознаны сканером (нечего сопоставлять с базой CVE).\n")

        f.write("\n=========================================================\n")
        f.write("Конец отчета.\n")

    print(f"[v] Отчет успешно сохранен.")


if __name__ == "__main__":
    run_scan_and_analyze()
