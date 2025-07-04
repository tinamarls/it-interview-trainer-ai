from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import ChromiumOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import logging
from urllib.parse import urlparse

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vacancy_parser")

# Кэш для URL
url_cache = {}

def extract_text_from_url(url, use_cache=False):
    """
    Извлекает релевантный текст вакансии с веб-страницы.

    Args:
        url: URL-адрес вакансии
        use_cache: использовать ли кэширование результатов

    Returns:
        Структурированный текст вакансии
    """
    # Проверяем кэш
    if use_cache and url in url_cache:
        logger.info(f"Используем кэшированную версию для URL: {url}")
        return url_cache[url]

    driver = None

    try:
        # Определяем тип сайта с вакансией для выбора метода парсинга
        domain = urlparse(url).netloc

        # Настраиваем Selenium
        options = ChromiumOptions()
        # options.add_argument('--headless')

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(20)

        driver.get(url)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Выбираем метод парсинга в зависимости от домена
        if "tinkoff.ru" in domain:
            vacancy_text = parse_tinkoff_vacancy(driver)
        elif "hh.ru" in domain:
            vacancy_text = parse_hh_vacancy(driver)
        elif "habr.com" in domain:
            vacancy_text = parse_habr_vacancy(driver)
        else:
            # Общий метод для неизвестных сайтов
            vacancy_text = parse_generic_vacancy(driver)

        # Очищаем и структурируем текст
        vacancy_text = clean_vacancy_text(vacancy_text)

        # Сохраняем в кэш
        if use_cache:
            url_cache[url] = vacancy_text

        return vacancy_text

    except Exception as e:
        logger.error(f"Ошибка при извлечении текста вакансии: {e}")
        return f"Ошибка при извлечении текста вакансии: {e}"

    finally:
        if driver:
            driver.quit()

def parse_tinkoff_vacancy(driver):
    """Парсер для вакансий Тинькофф с использованием BeautifulSoup"""
    try:
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")

        # Ищем заголовок
        title_element = soup.select_one("h1.header")
        title = title_element.get_text(strip=True) if title_element else "Не найден заголовок вакансии"

        # Определяем ключевые слова для разделов
        section_keywords = {
            "description": ["Описание", "О нас", "О проекте"],
            "responsibilities": ["Обязанности", "Задачи"],
            "requirements": ["Требования", "Навыки"],
            "benefits": ["Мы предлагаем", "Условия"]
        }

        full_text = f"{title}\n\n"
        found_sections = {}

        # Ищем все заголовки
        headers = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        current_section = None
        current_text = []

        for element in soup.find_all(True):
            if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                if current_section and current_text:
                    found_sections[current_section] = "\n".join(current_text).strip()
                    current_text = []
                # Проверяем, соответствует ли заголовок ключевому слову
                for section, keywords in section_keywords.items():
                    if any(keyword.lower() in element.get_text().lower() for keyword in keywords):
                        current_section = section
                        break
                else:
                    current_section = None
            elif current_section and element.get_text(strip=True):
                current_text.append(element.get_text(strip=True))

        # Сохраняем последний раздел
        if current_section and current_text:
            found_sections[current_section] = "\n".join(current_text).strip()

        # Формируем результат
        if "description" in found_sections:
            full_text += f"ОПИСАНИЕ:\n{found_sections['description']}\n\n"
        if "responsibilities" in found_sections:
            full_text += f"ОБЯЗАННОСТИ:\n{found_sections['responsibilities']}\n\n"
        if "requirements" in found_sections:
            full_text += f"ТРЕБОВАНИЯ:\n{found_sections['requirements']}\n\n"
        if "benefits" in found_sections:
            full_text += f"УСЛОВИЯ:\n{found_sections['benefits']}\n\n"

        return clean_vacancy_text(full_text)

    except Exception as e:
        logger.error(f"Ошибка при парсинге вакансии Тинькофф: {e}")
        return parse_generic_vacancy(driver)

def parse_hh_vacancy(driver):
    """Парсер для вакансий с hh.ru"""
    try:
        # Ищем блок с описанием вакансии
        vacancy_description = driver.find_element(By.CSS_SELECTOR, "div[data-qa='vacancy-description']")
        return vacancy_description.text
    except:
        return parse_generic_vacancy(driver)

def parse_habr_vacancy(driver):
    """Парсер для вакансий с Хабр Карьеры"""
    try:
        # Ищем блоки с информацией о вакансии
        description = driver.find_element(By.CSS_SELECTOR, "div.vacancy-description__text")
        return description.text
    except:
        return parse_generic_vacancy(driver)

def parse_generic_vacancy(driver):
    """Общий метод для парсинга вакансий с неизвестных сайтов"""

    # Пытаемся найти основные блоки вакансии по ключевым словам и HTML-структуре
    vacancy_text = ""

    # 1. Ищем по популярным CSS-селекторам
    selectors = [
        "div.vacancy-description",
        "div.job-description",
        "article.vacancy",
        "section.vacancy-details",
        "div.description",
        "div.vacancy-section"
    ]

    for selector in selectors:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        if elements:
            for element in elements:
                vacancy_text += element.text + "\n\n"
            return vacancy_text

    # 2. Если не нашли по селекторам, ищем по ключевым словам
    if not vacancy_text:
        sections = find_all_vacancy_sections(driver)
        if sections:
            return sections

    # 3. Если и это не сработало, применяем метод извлечения основного текста страницы
    return extract_main_content(driver)

def find_section_by_keywords(driver, keywords):
    """Находит раздел вакансии по ключевым словам в заголовках"""
    for keyword in keywords:
        # Ищем элементы с текстом, содержащим ключевое слово
        elements = driver.find_elements(By.XPATH, f"//*[contains(translate(text(), 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ', 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'), '{keyword.lower()}')]")
        for element in elements:
            tag_name = element.tag_name.lower()
            # Проверяем, является ли элемент заголовком
            if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                try:
                    # Собираем текст до следующего заголовка
                    text = ""
                    next_elements = element.find_elements(By.XPATH, "./following-sibling::*")
                    for sibling in next_elements:
                        sibling_tag = sibling.tag_name.lower()
                        # Прекращаем, если встретили новый заголовок
                        if sibling_tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                            break
                        # Игнорируем пустые элементы
                        if sibling.text.strip():
                            text += sibling.text + "\n"
                    return text.strip()
                except Exception as e:
                    logger.debug(f"Ошибка при извлечении текста для {keyword}: {e}")
                    continue
    return ""

def find_all_vacancy_sections(driver):
    """Находит все разделы вакансии по структуре страницы"""

    section_keywords = {
        "description": ["Описание", "О проекте", "О нас", "О компании", "О вакансии", "Description"],
        "responsibilities": ["Обязанности", "Задачи", "Что нужно делать", "Responsibilities", "What you'll do"],
        "requirements": ["Требования", "Навыки", "Что ожидаем", "Requirements", "Skills", "What you need"],
        "benefits": ["Мы предлагаем", "Условия", "Что вы получите", "Benefits", "What we offer", "Что мы предлагаем"]
    }

    found_sections = {}

    # Проходим по всем типам секций
    for section_type, keywords in section_keywords.items():
        section_text = find_section_by_keywords(driver, keywords)
        if section_text:
            found_sections[section_type] = section_text

    # Если нашли хотя бы одну секцию, формируем структурированный текст
    if found_sections:
        result = ""

        if "description" in found_sections:
            result += f"ОПИСАНИЕ:\n{found_sections['description']}\n\n"

        if "responsibilities" in found_sections:
            result += f"ОБЯЗАННОСТИ:\n{found_sections['responsibilities']}\n\n"

        if "requirements" in found_sections:
            result += f"ТРЕБОВАНИЯ:\n{found_sections['requirements']}\n\n"

        if "benefits" in found_sections:
            result += f"УСЛОВИЯ:\n{found_sections['benefits']}\n\n"

        return result

    return ""

def extract_main_content(driver):
    """Извлекает основной контент страницы, исключая навигацию, футер и т.д."""

    # Получаем HTML и создаем объект BeautifulSoup
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    # Удаляем ненужные элементы
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "meta", "link", "noscript", "iframe", "svg", "button"]):
        tag.extract()

    # Удаляем элементы с определенными классами/id
    noise_classes = ["footer", "header", "navigation", "menu", "sidebar",
                     "advertisement", "banner", "cookie", "popup", "modal",
                     "similar", "recommended", "related", "share", "social"]

    for noise_class in noise_classes:
        for element in soup.select(f"[class*='{noise_class}'], [id*='{noise_class}']"):
            element.extract()

    # Получаем текст страницы
    text = soup.get_text(separator=" ", strip=True)

    # Упрощенное извлечение основного контента страницы
    cleaned_text = clean_vacancy_text(text)

    return cleaned_text

def clean_vacancy_text(text):
    """Очищает текст вакансии от мусора и нормализует его"""
    # Удаляем лишние переносы строк и пробелы
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)

    # Список фраз-маркеров конца описания вакансии
    end_markers = [
        "Откликнуться на вакансию",
        "Похожие вакансии",
        "Отправить резюме",
        "Прикрепите файл",
        "Заполняя форму",
        "Отправить отклик",
        "Apply for this job",
        "Similar vacancies"
    ]

    # Обрезаем текст на первом из найденных маркеров
    for marker in end_markers:
        if marker in text:
            text = text.split(marker)[0]

    # Удаляем дублирующиеся заголовки и их содержимое
    sections = ["ОПИСАНИЕ:", "ОБЯЗАННОСТИ:", "ТРЕБОВАНИЯ:", "УСЛОВИЯ:"]
    cleaned_text = ""
    current_section = None
    for line in text.split("\n"):
        if line in sections:
            current_section = line
            cleaned_text += line + "\n"
        elif current_section and line.strip():
            # Проверяем, не начинается ли строка с ключевого слова другого раздела
            if any(line.startswith(s.strip(":")) for s in sections):
                continue
            cleaned_text += line + "\n"

    return cleaned_text.strip()