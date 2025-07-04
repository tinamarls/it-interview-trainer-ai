import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class ComparasionService:
    def __init__(self,
                 model_dir="/Users/kristina/IdeaProjects/it-train-diploma/machine-learning-service/app/services/my-skill-extractor-model",
                 resume_text=None, vacancy_text=None):
        self.minimal_score = 0.1
        self.model_dir = model_dir
        self.resume_text = resume_text
        self.vacancy_text = vacancy_text
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compare_texts(self):
        print("\n--- Использование дообученной модели для извлечения навыков ---")
        try:
            skill_tokenizer = AutoTokenizer.from_pretrained(f"{self.model_dir}/best-model")
            skill_model = AutoModelForTokenClassification.from_pretrained(f"{self.model_dir}/best-model").to(self.device)

            skill_pipeline = pipeline("ner", model=skill_model, tokenizer=skill_tokenizer, aggregation_strategy="simple",
                                      device=0 if self.device.type == "cuda" else -1)

            print(f"\nИзвлечение навыков из резюме: '{self.resume_text}'")
            extracted_skills_resume = skill_pipeline(self.resume_text)
            skills_list_resume = []
            for entity in extracted_skills_resume:
                if entity['entity_group'] == 'SKILL':
                    if entity['score'] > self.minimal_score:
                        skills_list_resume.append(entity['word'])
                        print("extracted_skill_resume:" + "word: " + entity['word'] + " " + str(entity['score']) + "")

            extracted_skills_vacancy = skill_pipeline(self.vacancy_text)
            skills_list_vacancy = []
            for entity in extracted_skills_vacancy:
                if entity['entity_group'] == 'SKILL':
                    if entity['score'] > self.minimal_score:
                        skills_list_vacancy.append(entity['word'])
                        print("extracted_skill_vacancy:" + "word: " + entity['word'] + "                     |" + str(entity['score']) + "")

            skills_list_resume = {skill.lower() for skill in skills_list_resume}
            skills_list_vacancy = {skill.lower() for skill in skills_list_vacancy}
            skills_list_vacancy_copy = skills_list_vacancy.copy()

            total_number = len(skills_list_vacancy)
            matches_number = 0

            for skill in skills_list_vacancy:
                if skill in skills_list_resume:
                    skills_list_vacancy_copy.remove(skill)
                    matches_number += 1
                    print("найдено совпадение: ", skill)



            return {
                "matching_percent": matches_number / total_number * 100,
                "needed_skills": str(skills_list_vacancy_copy)
            }

        except Exception as e:
            print(f"Ошибка при загрузке или использовании дообученной модели: {e}")
            print(f"Убедитесь, что модель была успешно обучена и сохранена в '{self.model_dir}/best-model'.")
            print("Если обучение не проводилось или было прервано, дообученная модель может быть недоступна.")

if __name__ == "__main__":
    resume_text = """
    Коржуева Кристина
Владимировна
Женщина, 20 лет, родилась 11 сентября 2003
+7 (987) 0379906 — предпочитаемый способ связи  •  можно обращаться также в telegram
@tinamarls
kkorzhuyeva11@mail.ru
Проживает: Казань
Гражданство: Россия, есть разрешение на работу: Россия
Не готова к переезду, готова к командировкам
Желаемая должность и зарплата
QA-инженер
Специализации:
—  Тестировщик
Занятость: полная занятость
График работы: полный день
Желательное время в пути до работы: не имеет значения
Опыт работы — 10 месяцев
Август 2023 —
настоящее время
10 месяцев
Центральный банк Российской Федерации
Казань, www.cbr.ru
Финансовый сектор
• Банк
Инженер 1 категории
Проведение ручного тестирования web-приложения, автоматизации тестирования согласно
тесткейсам в Jira с использованием Selenium и Java Selenide, нахождение и сопровождение
дефектов, составление тестовых сценариев в Jira, разбор логов, работа с эмулятора, ActiveMQ
Образование
Неоконченное высшее
2025 Казанский (Приволжский) федеральный университет, Казань
Институт информационных технологий и интеллектуальных систем, Программная инженерия
Повышение квалификации, курсы
2024 Академия Бэкенда
Тинькофф,  Java- разработка
Ключевые навыки
Знание языков Русский — Родной
Английский — B1 — Средний
Навыки
Резюме обновлено 7 мая 2024 в 07:17
 Java      Git      Atlassian Jira      PostgreSQL      Selenium      Ручное тестирование 
 QA      REST      Автоматизированное тестирование      API      Apache Maven 
 Selenide      HTTP      Postman      Spring Framework      ActiveMQ 
Дополнительная информация
Обо мне Предпочитаемый формат работы - гибрид, удаленный.
В данный момент учусь в бакалавриате по направлению "Программная инженерия". В
рамках учебной программы самостоятельно реализовывала проект с использованием
технологий Java Spring, html, css, js(https://github.com/oris-2sem/semester-work-tinamarls). Есть
базовые знания Python, которые использовала во время прохождения курса по DataMining.
Soft skills: ответственная, всегда рада учиться новому, не боюсь новых технологий.
Коржуева Кристина  •  Резюме обновлено 7 мая 2024 в 07:17

    """
    vacancy_text = """
        ОПИСАНИЕ:
Наша команда разрабатывает платформу, которая позволяет создавать ассистентов, используя современные внешние (Open AI) и внутренние LLM-модели. Это тип моделей глубокого обучения, которые понимают и генерируют текст на человеческом языке.
Наши пользователи — другие команды разработки, которые делают продукты в сегменте B2C. Наша модель уже помогла создать финансового ассистента, инвест-, шопинг-, тревел-ассистентов, Джуниора и Секретаря.
Вот какими проектами вы сможете заниматься:
— B2B-платформа LLM-ассистентов;
— ToolsRegistry: единый реестр инструментов, которые могут использовать LLM-модели в компании;
— централизованное решение для доступа к внешним и внутренним LLM-моделям.
Наш стек: Golang, K8s, helm, Gitlab, Postgres, Redis, Cassandra.
ОБЯЗАННОСТИ:
Разрабатывать нагруженные backend-приложения
Участвовать в проектировании внутренней микросервисной архитектуры приложений
Прорабатывать метрики приложения, которые помогут настроить мониторинг и алертинг по доступности
Поддерживать тестирование выполненных задач
ТРЕБОВАНИЯ:
Умеете разрабатывать многопоточные приложения
Знаете, как разрабатывать web-сервисы (REST, gRPC)
Умеете работать с реляционными базами данных, преимущественно с PostgreSQL
Уже работали с Unix-системами
Умеете работать с распределенными системами и очередями сообщений (Kafka)
Знаете, как работать с k8s и docker
Соблюдаете сроки
УСЛОВИЯ:
Работу в офисе или удаленно — по договоренности
Платформу обучения и развития «Апгрейд». Курсы, тренинги, вебинары и базы знаний. Поддержку менторов и наставников, помощь в поиске точек роста и карьерном развитии
Комплексную программу заботы о здоровье. Оформим полис ДМС с широким покрытием и страховку от несчастных случаев. Предложим льготные условия страхования для ваших близких
Возможность работы в аккредитованной ИТ-компании
Линейку льготных тарифов на продукты Т‑Банка
Частичную компенсацию затрат на спорт
Well-being-программу, которая помогает улучшить психологическое и физическое здоровье, а также разобраться с юридическими и финансовыми вопросами
Три дополнительных дня отпуска в год
Достойную зарплату — обсудим ее на собеседовании
    """
    cs = ComparasionService(resume_text=resume_text, vacancy_text=vacancy_text)
    cs.compare_texts()
