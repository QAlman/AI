from pathlib import Path
import re

# Укажите путь к нужной папке
folder_path = Path("путь/к/вашей/папке")
search_query = "искомое слово"  # текст, который нужно найти

CONTEXT_BEFORE = 3  # сколько слов до запроса
CONTEXT_AFTER = 5   # сколько слов после запроса


def get_context_with_lines(text: str, query: str, before: int, after: int):
    """
    Возвращает список кортежей (номер_строки, фрагмент_текста)
    для каждого вхождения запроса.
    """
    query_lower = query.lower()
    text_lower = text.lower()
    results = []
    start = 0

    while True:
        idx = text_lower.find(query_lower, start)
        if idx == -1:
            break

        end_idx = idx + len(query)

        # Номер строки = количество '\n' до позиции вхождения + 1
        line_number = text.count("\n", 0, idx) + 1

        # Контекст: берём символы до и после, затем отрезаем по словам
        before_text = text[:idx]
        after_text = text[end_idx:]

        words_before = re.findall(r"\S+", before_text)
        words_after = re.findall(r"\S+", after_text)

        prefix = " ".join(words_before[-before:]) if before else ""
        suffix = " ".join(words_after[:after]) if after else ""
        found = text[idx:end_idx]

        fragment = " ".join(part for part in [prefix, found, suffix] if part)
        results.append((line_number, fragment))

        start = end_idx

    return results


# rglob('*.txt') найдет все .txt файлы в папке и во всех её подпапках
for file_path in folder_path.rglob("*.txt"):
    try:
        content = file_path.read_text(encoding="utf-8")

        if search_query.lower() in content.lower():
            print("=" * 70)
            print(f"Файл: {file_path.name}")
            print(f"Путь: {file_path}")
            print("-" * 70)

            fragments = get_context_with_lines(content, search_query, CONTEXT_BEFORE, CONTEXT_AFTER)
            for i, (line_no, frag) in enumerate(fragments, 1):
                print(f"  [{i}] строка {line_no}: ...{frag}...")

    except Exception as e:
        print(f"Ошибка при чтении {file_path}: {e}")
