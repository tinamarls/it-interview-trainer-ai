from typing import Dict, Any, List, Tuple
# from soft_skills_analysis.utils.logger import logger # Убедитесь, что этот импорт нужен и корректен
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class OpenFaceDetector:
    def __init__(self):
        pass

    def calculate_blink_frequency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Подсчитывает частоту моргания на основе данных OpenFace AU45.
        """
        if df.empty:
            return {
                "total_blinks": 0,
                "blink_frequency_bpm": 0.0,
                "error": "DataFrame пуст"
            }

        # Проверяем наличие необходимых колонок
        if 'timestamp' not in df.columns or 'AU45_c' not in df.columns:
            missing_cols = [col for col in ['timestamp', 'AU45_c'] if col not in df.columns]
            return {
                "total_blinks": 0,
                "blink_frequency_bpm": 0.0,
                "error": f"В DataFrame отсутствуют необходимые колонки: {missing_cols}"
            }

        # Определяем моргание как переход от состояния не-моргания (AU45_c == 0)
        # к состоянию моргания (AU45_c == 1).
        # Используем .diff(), чтобы найти изменения между соседними кадрами.
        # Переход 0 -> 1 даст разницу +1.
        blink_starts = df['AU45_c'].diff()

        # Подсчитываем количество таких переходов (+1)
        # Игнорируем первый кадр, где .diff() будет NaN
        total_blinks = (blink_starts == 1).sum()

        # Вычисляем общую длительность видео по последней временной метке
        video_duration_seconds = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]

        # Избегаем деления на ноль
        if video_duration_seconds <= 0:
            return {
                "total_blinks": total_blinks, # Может быть 0, но это корректно для нулевой длительности
                "blink_frequency_bpm": 0.0,
                "error": "Длительность видео составляет 0 или отрицательное число."
            }

        # Переводим длительность в минуты
        video_duration_minutes = video_duration_seconds / 60.0

        # Вычисляем частоту моргания в морганиях в минуту
        blink_frequency_bpm = (total_blinks / video_duration_minutes) if video_duration_minutes > 0 else 0.0

        return {
            "total_blinks": int(total_blinks),
            "blink_frequency_bpm": round(blink_frequency_bpm, 2)
        }

    def calculate_smile_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализирует улыбку на основе данных OpenFace AU12 (подъем уголков губ).
        Возвращает информацию о продолжительности и проценте улыбки.
        """
        if df.empty:
            return {
                "total_smile_duration_seconds": 0,
                "smile_percentage": 0.0,
                "error": "DataFrame пуст"
            }

        # Вычисляем общую длительность видео
        video_duration_seconds = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]

        # Избегаем деления на ноль
        if video_duration_seconds <= 0:
            return {
                "total_smile_duration_seconds": 0,
                "smile_percentage": 0.0,
                "error": "Длительность видео составляет 0 или отрицательное число."
            }

        # Получаем кадры, где присутствует улыбка (AU12_c == 1)
        smile_frames = df[df['AU12_c'] == 1]

        # Подсчитываем количество кадров с улыбкой
        smile_frame_count = len(smile_frames)

        # Получаем общее количество кадров
        total_frames = len(df)

        # Вычисляем примерную продолжительность улыбки
        # Предполагаем, что кадры идут с равномерной частотой
        frame_duration = video_duration_seconds / total_frames
        total_smile_duration_seconds = smile_frame_count * frame_duration

        # Вычисляем процент времени с улыбкой
        smile_percentage = (smile_frame_count / total_frames) * 100.0

        return {
            "total_smile_duration_seconds": round(total_smile_duration_seconds, 2),
            "smile_percentage": round(smile_percentage, 2)
        }

    def calculate_gaze_aversion(self, df: pd.DataFrame, threshold_angle: float = 0.3, gaze_shift_threshold: float = 0.05) -> Dict[str, Any]:
        """
        Анализирует отведение взгляда и частоту "бегающего" взгляда на основе данных OpenFace.
        """
        required_cols = ['timestamp', 'gaze_angle_x', 'gaze_angle_y']
        video_duration_seconds = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        if df.empty or not all(col in df.columns for col in required_cols) or video_duration_seconds <= 0:
            return {
                "total_gaze_aversions": 0,
                "total_aversion_duration_seconds": 0,
                "aversion_percentage": 0.0,
                "aversion_frequency_per_minute": 0.0,
                "gaze_shift_frequency_per_minute": 0.0,
                "error": "DataFrame пуст"
            }

        total_frames = len(df)
        frame_duration = video_duration_seconds / total_frames
        video_duration_minutes = video_duration_seconds / 60.0

        # Расчёт отклонения взгляда от центра (модуль вектора)
        df['gaze_deviation'] = np.sqrt(df['gaze_angle_x']**2 + df['gaze_angle_y']**2)
        df['gaze_averted'] = df['gaze_deviation'] > threshold_angle

        # Кол-во кадров с отведенным взглядом
        averted_frame_count = df['gaze_averted'].sum()
        total_aversion_duration_seconds = averted_frame_count * frame_duration
        aversion_percentage = (averted_frame_count / total_frames) * 100.0

        # Переходы в отведение (0 → 1)
        gaze_aversion_starts = df['gaze_averted'].astype(int).diff()
        total_gaze_aversions = (gaze_aversion_starts == 1).sum()
        aversion_frequency_per_minute = (
                total_gaze_aversions / video_duration_minutes) if video_duration_minutes > 0 else 0.0

        # Оценка "бегающего" взгляда: насколько часто изменяется направление
        df['gaze_shift'] = np.sqrt(df['gaze_angle_x'].diff()**2 + df['gaze_angle_y'].diff()**2)
        gaze_shift_count = (df['gaze_shift'] > gaze_shift_threshold).sum()
        gaze_shift_frequency_per_minute = (gaze_shift_count / video_duration_minutes) if video_duration_minutes > 0 else 0.0

        return {
            "total_gaze_aversions": int(total_gaze_aversions),
            "total_aversion_duration_seconds": round(total_aversion_duration_seconds, 2),
            "aversion_percentage": round(aversion_percentage, 2),
            "aversion_frequency_per_minute": round(aversion_frequency_per_minute, 2),
            "gaze_shift_frequency_per_minute": round(gaze_shift_frequency_per_minute, 2)
        }




