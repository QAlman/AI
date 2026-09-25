import re
import json
import pypdf
from pathlib import Path

# Словарь для автоматической подстановки системных имен (Раздел 2.2)
# Сюда добавлены основные правила, которые чаще всего встречаются в аудите
USER_RIGHTS_MAP = {
    "Access this computer from the network": "SeNetworkLogonRight",
    "Act as part of the operating system": "SeTcbPrivilege",
    "Adjust memory quotas for a process": "SeIncreaseQuotaPrivilege",
    "Allow log on locally": "SeInteractiveLogonRight",
    "Allow log on through Remote Desktop Services": "SeRemoteInteractiveLogonRight",
    "Back up files and directories": "SeBackupPrivilege",
    "Bypass traverse checking": "SeChangeNotifyPrivilege",
    "Change the system time": "SeSystemtimePrivilege",
    "Change the time zone": "SeTimeZonePrivilege",
    "Create a pagefile": "SeCreatePagefilePrivilege",
    "Create a token object": "SeCreateTokenPrivilege",
    "Create global objects": "SeCreateGlobalPrivilege",
    "Create permanent shared objects": "SeCreatePermanentPrivilege",
    "Create symbolic links": "SeCreateSymbolicLinkPrivilege",
    "Debug programs": "SeDebugPrivilege",
    "Deny access to this computer from the network": "SeDenyNetworkLogonRight",
    "Deny log on as a batch job": "SeDenyBatchLogonRight",
    "Deny log on as a service": "SeDenyServiceLogonRight",
    "Deny log on locally": "SeDenyInteractiveLogonRight",
    "Deny log on through Remote Desktop Services": "SeDenyRemoteInteractiveLogonRight",
    "Enable computer and user accounts to be trusted for delegation": "SeEnableDelegationPrivilege",
    "Force shutdown from a remote system": "SeRemoteShutdownRight",
    "Generate security audits": "SeAuditPrivilege",
    "Impersonate a client after authentication": "SeImpersonatePrivilege",
    "Increase a process working set": "SeIncreaseWorkingSetPrivilege",
    "Load and unload device drivers": "SeLoadDriverPrivilege",
    "Lock pages in memory": "SeLockMemoryPrivilege",
    "Log on as a batch job": "SeBatchLogonRight",
    "Log on as a service": "SeServiceLogonRight",
    "Manage auditing and security log": "SeSecurityPrivilege",
    "Modify an object label": "SeRelabelPrivilege",
    "Modify firmware environment values": "SeSystemEnvironmentPrivilege",
    "Perform volume maintenance tasks": "SeManageVolumePrivilege",
    "Profile single process": "SeProfileSingleProcessPrivilege",
    "Profile system performance": "SeSystemProfilePrivilege",
    "Replace a process level token": "SeAssignPrimaryTokenPrivilege",
    "Restore files and directories": "SeRestorePrivilege",
    "Shut down the system": "SeShutdownPrivilege",
    "Take ownership of files or other objects": "SeTakeOwnershipPrivilege"
}


def extract_text_from_pdf(pdf_path):
    print(f"[INFO] Чтение PDF-файла: {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    full_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)
    return "\n".join(full_text)


def clean_text(text):
    if not text:
        return ""
    # Убираем лишние переносы строк внутри предложений, сохраняя абзацы
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Убираем множественные пробелы
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def parse_cis_text_to_json(full_text):
    print("[INFO] Анализ структуры и фильтрация оглавления...")

    # Ищем заголовки вроде "2.2.22 (L1) Ensure..."
    rule_start_pattern = re.compile(r"^(\d+(?:\.\d+)+)\s+\((L1|L2)[^\)]*\)\s+(.+)$", re.MULTILINE)
    matches = list(rule_start_pattern.finditer(full_text))
    benchmarks = []

    for i in range(len(matches)):
        match = matches[i]
        rule_id = match.group(1)
        profile = match.group(2)
        title = match.group(3).strip()

        # Фильтр оглавления: если в конце названия куча точек и номер страницы, пропускаем
        if re.search(r'\.{3,}\s*\d+$', title):
            continue

        # Вырезаем текст конкретного правила
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        rule_body = full_text[start_pos:end_pos]

        # Вытаскиваем блоки контента
        desc_m = re.search(r"Description:\s*(.*?)(?=Rationale:|Audit:|Remediation:|Default Value:|$)", rule_body,
                           re.DOTALL)
        rat_m = re.search(r"Rationale:\s*(.*?)(?=Audit:|Remediation:|Default Value:|$)", rule_body, re.DOTALL)
        audit_m = re.search(r"Audit:\s*(.*?)(?=Remediation:|Default Value:|$)", rule_body, re.DOTALL)
        rem_m = re.search(r"Remediation:\s*(.*?)(?=Default Value:|$)", rule_body, re.DOTALL)

        description = clean_text(desc_m.group(1)) if desc_m else ""
        audit_text = clean_text(audit_m.group(1)) if audit_m else ""
        remediation_text = clean_text(rem_m.group(1)) if rem_m else ""
        rationale = clean_text(rat_m.group(1)) if rat_m else ""

        # Фильтр: если описание и аудит пустые — это мусорный кусок (или дубль из оглавления), пропускаем
        if not description and not audit_text:
            continue

        rule_type = "unknown"
        tech_key = ""
        expected_value = ""
        match_type = "exact"

        # 1. ОПРЕДЕЛЯЕМ ПРАВА ПОЛЬЗОВАТЕЛЕЙ (Раздел 2.2)
        if "User Rights Assignment" in remediation_text or "Se" in audit_text:
            rule_type = "user_right"
            match_type = "contains"
            if "Guests" in title:
                expected_value = "*S-1-5-32-546"  # SID группы Гости

            # Ищем системное имя в словаре маппинга по названию правила
            for human_name, system_name in USER_RIGHTS_MAP.items():
                if human_name.lower() in title.lower():
                    tech_key = system_name
                    break
            if not tech_key:
                # Если в словаре нет, пытаемся вытащить регуляркой из текста
                se_match = re.search(r"(Se[A-Z][a-zA-Z]+Right|Se[A-Z][a-zA-Z]+Privilege)", rule_body)
                tech_key = se_match.group(1) if se_match else "UNKNOWN_SE_RIGHT"

        # 3. ОПРЕДЕЛЯЕМ ПОЛИТИКИ ПАРОЛЕЙ И БЛОКИРОВОК (Разделы 1.1 и 1.2)
        elif "Account Policies\\Password Policy" in remediation_text or "Account Lockout Policy" in remediation_text:
            rule_type = "account_policy"
            match_type = "exact"

            # Словарь соответствия для названий политик паролей
            account_map = {
                "Enforce password history": "PasswordHistorySize",
                "Maximum password age": "MaximumPasswordAge",
                "Minimum password age": "MinimumPasswordAge",
                "Minimum password length": "MinimumPasswordLength",
                "Password must meet complexity requirements": "PasswordComplexity",
                "Account lockout duration": "LockoutDuration",
                "Account lockout threshold": "LockoutBadCount",
                "Reset account lockout counter after": "ResetLockoutCount"
            }

            for human_name, system_key in account_map.items():
                if human_name.lower() in title.lower():
                    tech_key = system_key
                    break

            # Пытаемся вытащить ожидаемое значение (например, "24" из титула)
            val_match = re.search(r'is set to\s+[\'\"]?(\d+)', title, re.IGNORECASE)
            if val_match:
                expected_value = val_match.group(1)
            elif "enabled" in title.lower():
                expected_value = "1"
            elif "disabled" in title.lower():
                expected_value = "0"

        # 2. ОПРЕДЕЛЯЕМ РЕЕСТР (Разделы 18, 19 и др.)
        elif "HKLM" in audit_text or "reg query" in audit_text.lower():
            rule_type = "registry"
            # Вытаскиваем ветку реестра (например, HKLM\Software\Policies\...)
            reg_path_m = re.search(r'(HKLM\\[A-Za-z0-9\\[\]_\-]+)', audit_text)
            # Вытаскиваем имя параметра после ключа /v
            reg_val_m = re.search(r'/v\s+([A-Za-z0-9_\-]+)', audit_text)

            if reg_path_m and reg_val_m:
                tech_key = f"{reg_path_m.group(1)}\\{reg_val_m.group(1)}"
            elif reg_path_m:
                tech_key = reg_path_m.group(1)
            else:
                tech_key = "UNKNOWN_REG_KEY"

            # Попытка вытащить ожидаемое значение (часто пишется как "is set to 1" или "Enabled")
            if "enabled" in title.lower():
                expected_value = "1"
            elif "disabled" in title.lower():
                expected_value = "0"

        # Собираем готовый объект правила
        benchmark_rule = {
            "id": rule_id,
            "profile": profile,
            "title": title,
            "type": rule_type,
            "tech_key": tech_key,
            "expected_value": expected_value,
            "match_type": match_type,
            "metadata": {
                "description": description,
                "rationale": rationale,
                "audit_command": audit_text,
                "remediation": remediation_text
            }
        }
        benchmarks.append(benchmark_rule)

    return benchmarks


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    # Пути к папкам DATA согласно вашей структуре проекта
    pdf_filename = script_dir.parent / "DATA" / "CIS_Microsoft_Windows_Server_2012_R2_Benchmark_v3.0.0_ARCHIVE.pdf"
    json_filename = script_dir.parent / "DATA" / "cis_benchmarks.json"

    try:
        pdf_text = extract_text_from_pdf(str(pdf_filename))
        parsed_data = parse_cis_text_to_json(pdf_text)

        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=2, ensure_ascii=False)

        print(f"\n[SUCCESS] Спарсено чистых правил: {len(parsed_data)}")
        print(f"[INFO] Результат успешно сохранен в файл:\n{json_filename}")

    except FileNotFoundError:
        print(f"[ERROR] Не удалось найти PDF-файл по пути:\n{pdf_filename}")
