import sqlite3


def compare_versions(v1, v2):
    """Семантическое сравнение версий для SQLite"""
    if v1 is None or v2 is None:
        return 0

    def to_list(v):
        return [
            int(x) if x.isdigit() else x for x in v.replace("-", ".").split(".")
        ]

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
            if p1 < p2:
                return -1
            if p1 > p2:
                return 1
        else:
            if str(p1) < str(p2):
                return -1
            if str(p1) > str(p2):
                return 1
    return 0


def search_vulnerabilities_with_cwe(
    vendor, product, user_version, db_path="my_smart_nvd.db"
):
    """Умный поиск по базе данных с выводом CVE и CWE"""
    conn = sqlite3.connect(db_path)
    conn.create_function("COMPARE", 2, compare_versions)
    cursor = conn.cursor()

    # Добавили c.cwe_id в SELECT
    query = """
        SELECT DISTINCT c.cve_id, c.cwe_id, c.description 
        FROM cves c
        JOIN cpe_matches m ON c.cve_id = m.cve_id
        WHERE m.vendor = ? AND m.product = ? AND (
            m.exact_version = ? 
            OR (
                (m.v_start_inc IS NULL OR COMPARE(?, m.v_start_inc) >= 0) AND
                (m.v_start_exc IS NULL OR COMPARE(?, m.v_start_exc) > 0) AND
                (m.v_end_inc IS NULL   OR COMPARE(?, m.v_end_inc) <= 0) AND
                (m.v_end_exc IS NULL   OR COMPARE(?, m.v_end_exc) < 0) AND
                (m.v_start_inc IS NOT NULL OR m.v_start_exc IS NOT NULL OR m.v_end_inc IS NOT NULL OR m.v_end_exc IS NOT NULL)
            )
        )
    """

    cursor.execute(
        query,
        (
            vendor,
            product,
            user_version,
            user_version,
            user_version,
            user_version,
            user_version,
        ),
    )
    results = cursor.fetchall()

    print(
        f"🔍 Результаты для {vendor} {product} (версия {user_version}): Найдено {len(results)} CVE"
    )
    print("=" * 80)
    for cve_id, cwe_id, desc in results:
        # Корректно отображаем, если CWE отсутствует (бывает NULL)
        cwe_display = cwe_id if cwe_id else "Не указан"

        print(f"[{cve_id}] | CWE: {cwe_display}")
        print(f"Описание: {desc}\n")
        print("-" * 80)

    conn.close()


# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ДЛЯ 7-ZIP ---
# Переводим в нижний регистр, так как в базах NVD вендор и продукт обычно пишутся как "7-zip"
search_vulnerabilities_with_cwe(
    vendor="7-zip", product="7-zip", user_version="19.00"
)
