import librosa
from typing import Dict
import librosa
import numpy as np
import re
from typing import Dict

# todo:
# 1. Type-Token Ratio (лексическое разнообразие)
# 2. Темп речи
# 3. Filler Rate - Индекс речевой чистоты
# 4. Коэффициент хезитации - доля пауз в речи
# 5. F0 Variability (вариабельность основного тона)

def get_f0_variability(audio_path: str) -> float:
    """
    Вариабельность основного тона (F0): стандартное отклонение частоты
    """
    y, sr = librosa.load(audio_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []

    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_values.append(pitch)

    if len(pitch_values) < 2:
        return 0.0

    return float(np.std(pitch_values))


def get_filler_rate(transcript: str, duration_sec: float) -> float:
    """
    Filler Rate (слов-паразитов в минуту)

    Нормы:
    - <3.0: отлично
    - 3.0-6.0: норма для спонтанной речи
    - >6.0: требуется коррекция
    """
    filler_words = [
        # Междометия-паузы
        'ээ', 'мм', 'аа', 'э-э', 'м-м', 'а-а', 'эм', 'хм',
        'ну', 'вот', 'это', 'типа', 'как бы', 'значит',

        # Модальные частицы
        'вроде', 'вроде бы', 'как сказать', 'практически',
        'собственно', 'собственно говоря', 'походу', 'как-то так',

        # Аппроксиматоры
        'примерно', 'где-то', 'около', 'в районе', 'типо',

        # Интенсификаторы
        'прям', 'реально', 'вообще', 'абсолютно', 'типа того',

        # Дискурсивные маркеры
        'короче', 'короче говоря', 'в общем', 'в общем-то',
        'то есть', 'так сказать', 'ну типа', 'в принципе',
        'допустим', 'скажем', 'понимаешь', 'знаешь',

        # Современные разговорные
        'блин', 'чё', 'чё-то', 'капец', 'ешкин кот', 'ёпрст',
        'ясно', 'понятно', 'ну это', 'в натуре', 'респект',

        # Комбинации
        'ну вот', 'ну это', 'вот такое', 'как бы это',
        'вот именно', 'так вот', 'ну ладно', 'как его',

        # Фразовые клатчи
        'это самое', 'там всё', 'и всё такое', 'и тому подобное',
        'ну и вот', 'как бы там ни было', 'как говорится',

        # Англицизмы-паразиты
        'окей', 'вау', 'оу', 'имхо', 'лол',

        # Эвфемизмы
        'ё-моё', 'япона мать', 'твою ж мать', 'святые угодники'
    ]

    text = transcript.lower()
    filler_pattern = r'\b(' + '|'.join(re.escape(f) for f in filler_words) + r')\b'
    filler_count = len(re.findall(filler_pattern, text))

    duration_min = duration_sec / 60
    return filler_count / duration_min if duration_min > 0 else 0.0


class SpeechFeatureExtractor:

    def extract_features(self, audio_path: str, transcript: str, duration_s: float) -> Dict[str, float]:
        """
        Возвращает словарь всех метрик.
        :param audio_path: путь к аудиофайлу (.wav)
        :param transcript: транскрипт речи
        :param duration_s: длительность аудио в секундах
        """
        return {
            "ttr": self.compute_ttr(transcript),
            "speech_rate_wpm": self.speech_rate(transcript, duration_s),
            "filler_rate": filler_rate(transcript, duration_s),
            "hesitation_coeff": self.hesitation_coefficient(audio_path),
            "f0_variability": f0_variability(audio_path),
        }

    def compute_ttr(self, transcript: str) -> float:
        """
        Вычисляет Type-Token Ratio (TTR) - показатель лексического разнообразия.

        TTR = количество уникальных слов (types) / общее количество слов (tokens)

        Нормы:
        - 0.6-0.8: оптимальный диапазон для спонтанной речи
        - <0.5: низкое разнообразие, возможен ограниченный словарный запас
        - >0.9: неестественно высокое разнообразие (может указывать на заученный текст)
        """
        tokens = transcript.lower().split()
        if not tokens:
            return 0.0
        types = set(tokens)
        return len(types) / len(tokens)

    def hesitation_coefficient(self, audio_path: str) -> float:
        """
        Коэффициент хезитации (доля пауз в речи).
        Норма: <0.15, красный флаг: >0.25.
        """
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            intervals = librosa.effects.split(y, top_db=30, frame_length=2048)

            # Рассчет длительности пауз
            pause_duration = sum((end - start)/sr for start, end in intervals)
            total_duration = len(y)/sr

            return pause_duration / total_duration if total_duration > 0 else 0.0
        except Exception as e:
            print(f"Ошибка анализа пауз: {str(e)}")
            return 0.0

    def speech_rate(self, transcript: str, duration_s: float) -> float:
        """Темп речи: количество слов в минуту"""
        words = transcript.strip().split()
        if duration_s == 0:
            return 0.0
        return len(words) / (duration_s / 60)
