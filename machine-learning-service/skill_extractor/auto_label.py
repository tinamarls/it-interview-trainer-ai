import pandas as pd
import re
import os
import json
from sklearn.model_selection import train_test_split
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from datetime import datetime

# Пути
CSV_FILE_PATH = "/skill_extractor/data/skills/IT_vacancies_full.csv"
TEXT_COLUMN_NAME = 'description'
OUTPUT_COLUMN_NAME_TOKENS = 'tokens'
OUTPUT_COLUMN_NAME_BIO_TAGS = 'bio_tags'
SKILLS_PATH = "/skill_extractor/data/skills2.csv"
OUTPUT_DIR = "/data/labeled_text"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Загрузка данных
df = pd.read_csv(CSV_FILE_PATH)
skills_df = pd.read_csv(SKILLS_PATH)
uploaded_skills_text = skills_df['skill'].dropna().str.strip().str.lower().unique().tolist()
# Обработка загруженного текста навыков
processed_uploaded_skills = []
for line in uploaded_skills_text.splitlines():
    skill = line.strip().lower()
    if skill:
        ignored_keywords = [
            "abstract", "advertising", "analysis", "analytical skills", "bonusengine",
            "budgeting", "business analysis", "client management", "client success manager",
            "coaching", "conflict resolution", "conceptual design", "editing",
            "educational content", "hr", "hse", "interpersonal skills", "leadership skills",
            "legacy", "legacy code", "online courses", "online trainings", "problem-solving",
            "project management", "proposal", "reviewing", "roadmap", "team lead",
            "technical writer", "technical writing", "trade marketing", "talent acquisition",
            "1:", "1-", "/-", "++", "#", "class.php", "c++ boost", "code-review", "ode review",
            "onfluence", "-sql", "ip", "it", "qa", "unux", "windovs", "++11", "++14",
            "lickhouse", "vps/vds", "ubernetes", "aven", "osi", "crm-", "web-", "web ",
            "86", "path", "cel", "c ", "1c" # "1c" оставляем, "1c:" -> "1c"
        ]
        is_ignored = False
        for ignored_word in ignored_keywords:
            if ignored_word == skill or skill.startswith(ignored_word + " ") or (" " + ignored_word) in skill :
                if ignored_word in ["api", "web", "ui", "ux", "data", "qa", "it", "ip"] and len(skill.split()) > 1:
                    continue
                is_ignored = True
                break
        if not is_ignored:
            if skill == "1c:":
                skill = "1c"
            processed_uploaded_skills.append(skill)

# Объединение и удаление дубликатов
combined_skills_list = list(set(initial_skills + processed_uploaded_skills))

# Финальная сортировка: сначала самые длинные навыки
SKILLS_LIST = sorted(combined_skills_list, key=len, reverse=True)
# Удаление пустых строк на всякий случай, если они как-то попали
SKILLS_LIST = [s for s in SKILLS_LIST if s.strip()]


def tokenize_text(text):
    """
    Простая токенизация текста по пробелам и знакам препинания.
    Можно заменить на более сложный токенизатор (NLTK, spaCy), если требуется.
    """
    if not isinstance(text, str):
        return []
    text = re.sub(r'[,\.;:\(\)\[\]"\'!?«»]', ' ', text) # Основные разделители
    text = re.sub(r'\s+/\s+', ' / ', text) # Обработка / как возможного разделителя или части (ux/ui)

    raw_tokens = re.findall(r'[а-яА-Яa-zA-Z0-9\._\+\#\-/]+|[^\w\s]', text)

    tokens = [token for token in raw_tokens if token.strip() and not token.isspace()]
    return tokens


def bio_tag_vacancy(text, skills_dict):
    """
    Размечает текст вакансии BIO-тегами на основе списка навыков.
    skills_dict - это словарь {skill_lower: original_skill_tokens_count}
    """
    if not isinstance(text, str) or not text.strip():
        return [], []

    original_tokens = tokenize_text(text)
    if not original_tokens:
        return [], []

    tags = ['O'] * len(original_tokens)

    lower_original_tokens = [t.lower() for t in original_tokens]

    # Проходим по навыкам (уже отсортированы по длине от длинных к коротким)
    for skill_lower, num_skill_tokens in skills_dict.items():
        skill_parts = skill_lower.split() # Навык, разбитый на части по пробелам

        # Ищем вхождения навыка в токенизированном тексте вакансии
        i = 0
        while i <= len(lower_original_tokens) - num_skill_tokens:
            match = True
            already_tagged_in_span = False
            for k in range(num_skill_tokens):
                if tags[i+k] != 'O':
                    already_tagged_in_span = True
                    break
            if already_tagged_in_span:
                i += 1
                continue

            current_vacancy_tokens_segment = lower_original_tokens[i : i + num_skill_tokens]

            norm_skill = "".join(skill_parts).replace('.', '').replace('-', '')
            norm_segment = "".join(current_vacancy_tokens_segment).replace('.', '').replace('-', '')


            if norm_skill == norm_segment:
                pass
            elif skill_parts == current_vacancy_tokens_segment :
                pass
            else:
                match = False

            if match:
                tags[i] = 'B-SKILL'
                for j in range(1, num_skill_tokens):
                    tags[i+j] = 'I-SKILL'
                i += num_skill_tokens
            else:
                if num_skill_tokens == 1 and i < len(lower_original_tokens) :
                    current_token_lower = lower_original_tokens[i]
                    normalized_skill_single = skill_lower.replace('.', '').replace('+', 'plus').replace('#', 'sharp')
                    normalized_token_single = current_token_lower.replace('.', '').replace('+', 'plus').replace('#', 'sharp')

                    if normalized_skill_single == normalized_token_single and tags[i] == 'O':
                        tags[i] = 'B-SKILL'
                i += 1

    return original_tokens, tags


# --- Основной скрипт ---
if __name__ == "__main__":
    try:
        df = pd.read_csv(CSV_FILE_PATH).head(5000)
    except FileNotFoundError:
        print(f"Ошибка: Файл {CSV_FILE_PATH} не найден.")
        exit()
    except Exception as e:
        print(f"Ошибка при чтении CSV файла: {e}")
        exit()

    if TEXT_COLUMN_NAME not in df.columns:
        print(f"Ошибка: Колонка '{TEXT_COLUMN_NAME}' не найдена в CSV файле.")
        print(f"Доступные колонки: {df.columns.tolist()}")
        exit()

    # Подготавливаем словарь навыков для функции bio_tag_vacancy
    # Ключ - навык в нижнем регистре, значение - количество токенов в нем (по пробелам)
    # SKILLS_LIST уже отсортирован от длинных к коротким
    skills_dict_for_tagging = {skill.lower(): len(skill.split()) for skill in SKILLS_LIST}


    results = []
    print(f"Начинается обработка {len(df)} вакансий...")
    for index, row in df.iterrows():
        vacancy_text = row[TEXT_COLUMN_NAME]

        # Проверка на NaN или другие нестроковые типы
        if not isinstance(vacancy_text, str):
            vacancy_text = "" # Обрабатываем как пустую строку

        tokens, bio_tags = bio_tag_vacancy(vacancy_text, skills_dict_for_tagging)
        results.append({
            'original_text': vacancy_text if isinstance(vacancy_text, str) else "",
            OUTPUT_COLUMN_NAME_TOKENS: tokens,
            OUTPUT_COLUMN_NAME_BIO_TAGS: bio_tags
        })
        if (index + 1) % 50 == 0 or (index + 1) == len(df): # Выводить прогресс каждые 50 вакансий и в конце
            print(f"Обработано {index + 1}/{len(df)} вакансий...")


    output_df = pd.DataFrame(results)

    # Вывод или сохранение результата
    print("\n--- Результат BIO-разметки (первые 5 строк) ---")
    for i, row_res in output_df.head().iterrows():
        print(f"\nВакансия {i+1}:")
        print("Размеченные токены:")
        # Проверяем, что длины списков токенов и тегов совпадают
        if len(row_res[OUTPUT_COLUMN_NAME_TOKENS]) == len(row_res[OUTPUT_COLUMN_NAME_BIO_TAGS]):
            for token, tag in zip(row_res[OUTPUT_COLUMN_NAME_TOKENS], row_res[OUTPUT_COLUMN_NAME_BIO_TAGS]):
                pass
        else:
            print("Ошибка: количество токенов не совпадает с количеством тегов.")



    output_csv_file = 'vacancies_bio_tagged.csv'
    try:
        output_df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        print(f"\nРезультаты сохранены в файл: {output_csv_file}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов в CSV: {e}")