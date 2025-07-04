import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from soft_skills_analysis.audio_analysis.speech_features import SpeechFeatureExtractor

logger = logging.getLogger(__name__)


class SpeechAnalyzer:
    """
    Класс для анализа речи и предоставления комплексной оценки
    на основе различных характеристик речи.
    """

    def __init__(self):
        """Инициализация анализатора речи."""
        self.feature_extractor = SpeechFeatureExtractor()

        # Веса для расчета итогового скора
        self.weights = {
            "ttr": 0.2,  # Лексическое разнообразие
            "speech_rate_wpm": 0.2,  # Темп речи
            "filler_rate": 0.2,  # Слова-паразиты
            "hesitation_coeff": 0.2,  # Коэффициент неуверенности
            "f0_variability": 0.2,  # Вариативность интонации
        }

        # Оптимальные значения для каждой метрики
        self.optimal_values = {
            "ttr": 0.65,  # Хорошее лексическое разнообразие
            "speech_rate_wpm": 140,  # Оптимальный темп речи (слов в минуту)
            "filler_rate": 0.02,  # Низкий уровень слов-паразитов
            "hesitation_coeff": 0.1,  # Низкий уровень неуверенности
            "f0_variability": 0.4,  # Хорошая вариативность интонации
        }

        # Допустимые отклонения для каждой метрики
        self.deviations = {
            "ttr": 0.2,  # Допустимое отклонение для TTR
            "speech_rate_wpm": 40,  # Допустимое отклонение для темпа речи
            "filler_rate": 0.05,  # Допустимое отклонение для слов-паразитов
            "hesitation_coeff": 0.1,  # Допустимое отклонение для неуверенности
            "f0_variability": 0.2,  # Допустимое отклонение для вариативности интонации
        }

    def analyze_speech(self, audio_path: str, transcript: str, duration_s: float) -> Dict[str, List[Dict[str, Any]]]:
        """
        Анализирует аудио и возвращает список метрик оценки речи.

        Args:
            audio_path: Путь к аудиофайлу (.wav)
            transcript: Транскрипт речи
            duration_s: Длительность аудио в секундах

        Returns:
            Dict[str, List[Dict[str, Any]]]: Словарь со списком метрик
        """
        # Извлечение речевых характеристик
        try:
            speech_features = self.feature_extractor.extract_features(audio_path, transcript, duration_s)
            logger.info(f"Извлеченные характеристики речи: {speech_features}")
        except Exception as e:
            logger.error(f"Ошибка при извлечении характеристик речи: {e}")
            raise

        # Расчет скоров по каждой метрике
        feature_scores = self._calculate_feature_scores(speech_features)

        # Получаем интерпретации результатов
        interpretations = self._interpret_results(speech_features, feature_scores)

        # Формируем список метрик с их характеристиками
        metrics = []

        # Добавляем каждую метрику как отдельный объект в список
        for feature_name in speech_features:
            metric = {
                "name": self._get_feature_display_name(feature_name),
                "key": feature_name,
                "value": speech_features[feature_name],
                "score": feature_scores[feature_name],
                "interpretation": interpretations[feature_name],
                "recommendation": self._get_specific_recommendation(feature_name, speech_features[feature_name])
            }

            metrics.append(metric)

        return {
            "metrics": metrics
        }

    def _get_feature_display_name(self, feature_key: str) -> str:
        """Возвращает читаемое название для метрики"""
        display_names = {
            "ttr": "Лексическое разнообразие",
            "speech_rate_wpm": "Темп речи",
            "filler_rate": "Слова-паразиты",
            "hesitation_coeff": "Паузы неуверенности",
            "f0_variability": "Интонационная выразительность"
        }
        return display_names.get(feature_key, feature_key)

    def _get_specific_recommendation(self, feature_name: str, value: float) -> str:
        """Возвращает конкретную рекомендацию для отдельной метрики"""
        if feature_name == "ttr":
            if value > 0.9:
                return "Ваше лексическое разнообразие отличное, продолжайте в том же духе"
            elif value > 0.7:
                return "Хорошее лексическое разнообразие, можно немного расширить словарный запас"
            else:
                return "Старайтесь использовать более разнообразную лексику"

        elif feature_name == "speech_rate_wpm":
            if value < 100:
                return "Увеличьте темп речи, чтобы звучать более уверенно"
            elif value > 180:
                return "Снизьте темп речи для лучшего восприятия слушателями"
            else:
                return ""


    def _calculate_feature_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Рассчитывает скоры для каждой метрики в диапазоне от 0 до 1.

        Args:
            features: Словарь извлеченных характеристик речи

        Returns:
            Dict[str, float]: Словарь со скорами по каждой метрике
        """
        scores = {}

        for feature, value in features.items():
            optimal = self.optimal_values[feature]
            deviation = self.deviations[feature]

            if feature in ["ttr", "f0_variability"]:
                # Для этих метрик высокие значения лучше (но не слишком высокие)
                if value > optimal + deviation:
                    # Если значение слишком высокое - небольшой штраф
                    scores[feature] = 0.9 - min(0.4, (value - optimal - deviation) / deviation)
                elif value < optimal - deviation:
                    # Если значение слишком низкое - более строгий штраф
                    scores[feature] = max(0.1, 0.7 * (value / (optimal - deviation)))
                else:
                    # Если значение в оптимальном диапазоне
                    distance = abs(value - optimal) / deviation
                    scores[feature] = 1.0 - 0.3 * distance

            elif feature == "speech_rate_wpm":
                # Для темпа речи оптимально среднее значение
                distance = abs(value - optimal) / deviation
                scores[feature] = max(0.1, 1.0 - 0.7 * min(1.0, distance))

            elif feature in ["filler_rate", "hesitation_coeff"]:
                # Для этих метрик низкие значения лучше
                if value <= optimal:
                    scores[feature] = 1.0
                else:
                    # Если значение выше оптимального - штраф
                    distance = (value - optimal) / deviation
                    scores[feature] = max(0.1, 1.0 - 0.7 * min(1.0, distance))

        return scores

    def _calculate_overall_score(self, feature_scores: Dict[str, float]) -> float:
        """
        Рассчитывает итоговый скор на основе скоров по отдельным метрикам и их весов.

        Args:
            feature_scores: Словарь со скорами по каждой метрике

        Returns:
            float: Итоговый скор от 0 до 10
        """
        weighted_sum = 0
        total_weight = 0

        for feature, score in feature_scores.items():
            weight = self.weights.get(feature, 0)
            weighted_sum += score * weight
            total_weight += weight

        # Нормализация итогового скора от 0 до 10
        if total_weight > 0:
            normalized_score = (weighted_sum / total_weight) * 10
        else:
            normalized_score = 0

        return round(normalized_score, 1)

    def _interpret_results(self, features: Dict[str, float], scores: Dict[str, float]) -> Dict[str, str]:
        """
        Интерпретирует результаты анализа речи.

        Args:
            features: Словарь извлеченных характеристик речи
            scores: Словарь со скорами по каждой метрике

        Returns:
            Dict[str, str]: Словарь с интерпретациями для каждой метрики
        """
        interpretations = {}

        # Лексическое разнообразие (TTR)
        ttr = features.get("ttr", 0)
        if ttr < 0.4:
            interpretations[
                "ttr"] = "Очень низкое лексическое разнообразие. Речь монотонна и используется ограниченный набор слов."
        elif ttr < 0.5:
            interpretations["ttr"] = "Низкое лексическое разнообразие. Словарный запас требует расширения."
        elif ttr < 0.6:
            interpretations["ttr"] = "Среднее лексическое разнообразие."
        elif ttr < 0.7:
            interpretations["ttr"] = "Хорошее лексическое разнообразие."
        else:
            interpretations["ttr"] = "Отличное лексическое разнообразие. Богатый словарный запас."

        # Темп речи
        rate = features.get("speech_rate_wpm", 0)
        if rate < 100:
            interpretations["speech_rate_wpm"] = "Медленный темп речи, может восприниматься как неуверенность."
        elif rate < 120:
            interpretations["speech_rate_wpm"] = "Умеренно-медленный темп речи."
        elif rate < 160:
            interpretations["speech_rate_wpm"] = "Оптимальный темп речи для восприятия."
        elif rate < 180:
            interpretations["speech_rate_wpm"] = "Быстрый темп речи, но всё ещё хорошо воспринимаемый."
        else:
            interpretations["speech_rate_wpm"] = "Очень быстрый темп речи, может затруднять понимание."

        # Слова-паразиты
        filler = features.get("filler_rate", 0)
        if filler < 0.02:
            interpretations["filler_rate"] = "Минимальное количество слов-паразитов. Речь чистая и уверенная."
        elif filler < 0.05:
            interpretations[
                "filler_rate"] = "Небольшое количество слов-паразитов. Речь воспринимается как достаточно уверенная."
        elif filler < 0.1:
            interpretations[
                "filler_rate"] = "Заметное количество слов-паразитов. Речь воспринимается как недостаточно уверенная."
        else:
            interpretations[
                "filler_rate"] = "Высокое количество слов-паразитов. Речь может восприниматься как неподготовленная или неуверенная."

        # Неуверенность
        hesitation = features.get("hesitation_coeff", 0)
        if hesitation < 0.1:
            interpretations["hesitation_coeff"] = "Минимальное количество пауз неуверенности. Речь плавная и уверенная."
        elif hesitation < 0.2:
            interpretations[
                "hesitation_coeff"] = "Небольшое количество пауз неуверенности. Речь воспринимается как достаточно уверенная."
        elif hesitation < 0.3:
            interpretations[
                "hesitation_coeff"] = "Заметное количество пауз неуверенности. Речь может восприниматься как недостаточно уверенная."
        else:
            interpretations[
                "hesitation_coeff"] = "Высокое количество пауз неуверенности. Речь воспринимается как неуверенная."

        # Вариативность интонации
        f0_var = features.get("f0_variability", 0)
        if f0_var < 0.2:
            interpretations[
                "f0_variability"] = "Низкая вариативность интонации. Речь монотонная, может вызывать потерю внимания."
        elif f0_var < 0.3:
            interpretations["f0_variability"] = "Умеренная вариативность интонации."
        elif f0_var < 0.5:
            interpretations[
                "f0_variability"] = "Хорошая вариативность интонации. Речь выразительная и интересная для восприятия."
        else:
            interpretations[
                "f0_variability"] = "Высокая вариативность интонации. Речь очень выразительная и эмоциональная."

        return interpretations

    def _get_recommendations(self, features: Dict[str, float]) -> List[str]:
        """
        Генерирует рекомендации для улучшения речи на основе извлеченных характеристик.

        Args:
            features: Словарь извлеченных характеристик речи

        Returns:
            List[str]: Список рекомендаций
        """
        recommendations = []

        # Рекомендации по лексическому разнообразию
        ttr = features.get("ttr", 0)
        if ttr < 0.5:
            recommendations.append("Расширьте свой словарный запас, избегайте повторений одних и тех же слов.")

        # Рекомендации по темпу речи
        rate = features.get("speech_rate_wpm", 0)
        if rate < 120:
            recommendations.append("Увеличьте темп речи, чтобы звучать более уверенно.")
        elif rate > 160:
            recommendations.append("Снизьте темп речи для лучшего восприятия слушателями.")

        # Рекомендации по словам-паразитам
        filler = features.get("filler_rate", 0)
        if filler > 0.05:
            recommendations.append("Сократите использование слов-паразитов ('э-э', 'так', 'как бы', 'значит' и т.д.).")

        # Рекомендации по неуверенности
        hesitation = features.get("hesitation_coeff", 0)
        if hesitation > 0.2:
            recommendations.append(
                "Работайте над уменьшением количества пауз неуверенности, практикуйтесь в публичных выступлениях.")

        # Рекомендации по вариативности интонации
        f0_var = features.get("f0_variability", 0)
        if f0_var < 0.3:
            recommendations.append("Добавьте больше выразительности и вариативности в интонацию речи.")

        # Если все хорошо
        if not recommendations:
            recommendations.append(
                "Ваша речь хорошо сбалансирована. Продолжайте практиковаться для поддержания высокого уровня.")

        return recommendations
