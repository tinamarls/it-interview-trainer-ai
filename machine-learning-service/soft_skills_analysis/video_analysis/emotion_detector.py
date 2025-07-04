import os
import time
import cv2
import numpy as np
from fer import FER
from typing import List, Tuple, Dict, Optional, Any


class VideoEmotionDetector:
    """
    Класс для определения эмоций на видео.
    Анализирует эмоции на лицах в видеокадрах и возвращает
    последовательность доминирующих эмоций с соответствующими временными метками.
    """

    def __init__(self, use_mtcnn: bool = True, debug: bool = False):
        """
        Инициализация детектора эмоций.

        Args:
            use_mtcnn: Использовать ли MTCNN для определения лиц (более точный, но медленнее)
            debug: Режим отладки
        """
        self.detector = FER(mtcnn=use_mtcnn)
        self.debug = debug
        self.video_processor = None # Это поле, похоже, не используется в текущем классе

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Анализирует видео, извлекает последовательность эмоций, временные метки
        и вычисляет распределение эмоций, преобладающую эмоцию и частоту смены эмоций.

        Args:
            video_path: Путь к видеофайлу.

        Returns:
            Dict: Словарь, содержащий:
                - 'emotion_sequence': Список обнаруженных эмоций (List[str]).
                - 'timestamps': Список временных меток для каждой эмоции (List[float]).
                - 'dominant_emotion': Наиболее часто встречающаяся эмоция (str).
                - 'emotion_distribution': Словарь с распределением эмоций (Dict[str, float]),
                                          где ключ - эмоция, значение - ее доля.
                - 'average_emotions': То же, что и 'emotion_distribution' (Dict[str, float]).
                - 'emotion_change_frequency': Частота смены эмоций в минуту (float).
                - 'num_emotion_changes': Общее количество смен эмоций (int).
                - 'processing_details': Информация о процессе обработки (Dict[str, Any]).
                - 'error_message': Сообщение об ошибке, если возникла (str, опционально).
                  В случае ошибки другие поля могут отсутствовать или быть пустыми.
        """
        if not os.path.exists(video_path):
            print(f"Файл видео не найден: {video_path}")
            return {'error_message': f"Файл видео не найден: {video_path}"}

        emotion_sequence = []
        timestamps = []

        # Инициализация полей для возвращаемого словаря
        dominant_emotion_val = 'unknown'
        emotion_distribution_val = {}
        average_emotions_val = {}
        emotion_change_frequency_val = 0.0
        num_emotion_changes_val = 0

        processing_details = {}
        current_frame_index = 0
        processed_for_emotion_count = 0
        cap = None # Инициализация cap

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Ошибка: не удалось открыть видеофайл {video_path}")
                if cap: cap.release()
                return {'error_message': f"Не удалось открыть видеофайл {video_path}"}

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_count_info = int(frame_count_raw) if frame_count_raw > 0 else 0

            if fps <= 0:
                print(f"Предупреждение: невозможно определить FPS для видео {video_path}. Используется значение по умолчанию 25 FPS.")
                fps = 25

            skip_frames = max(1, int(round(fps)))

            processing_details['video_fps_reported'] = cap.get(cv2.CAP_PROP_FPS) # исходный FPS
            processing_details['video_fps_used'] = fps # используемый FPS
            processing_details['frame_count_reported'] = frame_count_info
            processing_details['skip_frames_setting'] = skip_frames

            if hasattr(self, 'debug') and self.debug: # Проверка наличия self.debug
                print(f"Video FPS: {fps:.2f} (используется для расчета timestamp).")
                if frame_count_info > 0:
                    print(f"Total frames (по данным видео): {frame_count_info}.")
                else:
                    print(f"Total frames (по данным видео): информация недоступна или некорректна (получено {frame_count_raw}).")
                print(f"Будет обрабатываться примерно каждый {skip_frames}-й кадр (цель: ~1 анализируемый кадр/сек).")

            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame_index % skip_frames == 0:
                    timestamp = current_frame_index / fps
                    detected_emotion = self.detect_emotion(frame)
                    emotion_sequence.append(detected_emotion)
                    timestamps.append(timestamp)
                    processed_for_emotion_count += 1

                    if hasattr(self, 'debug') and self.debug and (processed_for_emotion_count % 10 == 0) and emotion_sequence:
                        elapsed = time.time() - start_time
                        progress_msg = (
                            f"Обработано для эмоций: {processed_for_emotion_count} кадров. "
                            f"Текущее время видео: {timestamp:.1f}с. Затрачено времени: {elapsed:.1f}с."
                        )
                        if frame_count_info > 0 and skip_frames > 0:
                            estimated_total_analysis_frames = frame_count_info // skip_frames
                            if estimated_total_analysis_frames > 0:
                                progress_percent = (processed_for_emotion_count / estimated_total_analysis_frames) * 100
                                progress_msg += f" Предполагаемый прогресс: {progress_percent:.1f}%."
                        print(progress_msg)

                current_frame_index += 1

            cap.release()
            cap = None # Указываем, что ресурс освобожден

            processing_details['total_frames_read_from_video'] = current_frame_index
            processing_details['frames_processed_for_emotion'] = processed_for_emotion_count

            if emotion_sequence: # Если были обнаружены эмоции
                if hasattr(self, 'smooth_neutral_spikes'): # Проверка наличия метода
                    emotion_sequence = self.smooth_neutral_spikes(emotion_sequence)

                # 1. Вычисление emotion_distribution и dominant_emotion
                emotion_counts = {emotion: emotion_sequence.count(emotion) for emotion in set(emotion_sequence)}
                total_detected_emotions = len(emotion_sequence)
                if total_detected_emotions > 0:
                    emotion_distribution_val = {
                        emotion: count / total_detected_emotions
                        for emotion, count in emotion_counts.items()
                    }
                    if emotion_distribution_val:
                        dominant_emotion_val = max(emotion_distribution_val, key=emotion_distribution_val.get)

                # 2. average_emotions (принимаем таким же, как distribution)
                average_emotions_val = emotion_distribution_val

                # 3. Вычисление emotion_change_frequency
                if hasattr(self, 'calculate_emotion_changes'): # Проверка наличия метода
                    num_emotion_changes_val = self.calculate_emotion_changes(emotion_sequence)
                else: # Базовая реализация, если метод отсутствует
                    num_emotion_changes_val = 0
                    for i in range(len(emotion_sequence) - 1):
                        if emotion_sequence[i] != emotion_sequence[i+1]:
                            num_emotion_changes_val +=1

                processing_details['num_emotion_changes_raw'] = num_emotion_changes_val

                if timestamps and len(timestamps) > 1:
                    analyzed_duration_seconds = timestamps[-1] - timestamps[0]
                    if analyzed_duration_seconds > 0:
                        emotion_change_frequency_val = (num_emotion_changes_val / analyzed_duration_seconds) * 60 # в минуту
                    else: # Если длительность 0, но есть изменения (маловероятно с >1 timestamp)
                        emotion_change_frequency_val = 0.0 if num_emotion_changes_val == 0 else float('inf') # или обработать как ошибку
                elif timestamps and len(timestamps) <= 1 : # 0 или 1 эмоция, нет изменений между разными временными метками
                    emotion_change_frequency_val = 0.0
                # Если timestamps пуст, но emotions есть (не должно происходить по логике выше), частота остается 0.0

            else: # Если эмоции не были обнаружены
                print(f"Предупреждение: не удалось определить эмоции в видео {video_path}. Возможно, видео слишком короткое или лица не распознаны.")
                processing_details['notes'] = "Эмоции не обнаружены."

            total_processing_time = time.time() - start_time
            processing_details['total_processing_time_seconds'] = round(total_processing_time, 2)

            if hasattr(self, 'debug') and self.debug:
                print(f"Обнаружено {len(emotion_sequence)} эмоций (ключевых кадров):")
                if emotion_distribution_val:
                    for emotion, dist_val in sorted(emotion_distribution_val.items(), key=lambda x: x[1], reverse=True):
                        print(f"  - {emotion}: {emotion_sequence.count(emotion)} ({dist_val:.1%})")
                print(f"Преобладающая эмоция: {dominant_emotion_val}")
                print(f"Количество изменений эмоций: {num_emotion_changes_val}")
                print(f"Частота смены эмоций (в минуту): {emotion_change_frequency_val:.2f}")
                print(f"Общее время обработки: {total_processing_time:.1f} с.")

            return {
                'emotion_sequence': emotion_sequence,
                'timestamps': timestamps,
                'dominant_emotion': dominant_emotion_val,
                'emotion_distribution': emotion_distribution_val,
                'average_emotions': average_emotions_val, # Копия distribution
                'emotion_change_frequency': round(emotion_change_frequency_val, 2),
                'num_emotion_changes': num_emotion_changes_val,
                'processing_details': processing_details
            }

        except Exception as e:
            print(f"Критическая ошибка при анализе видео {video_path}: {e}")
            import traceback # для более детального лога ошибки
            print(traceback.format_exc())
            if cap is not None and cap.isOpened(): # Убедимся, что cap существует и открыт перед release
                cap.release()
            return {
                'emotion_sequence': [],
                'timestamps': [],
                'dominant_emotion': 'error',
                'emotion_distribution': {},
                'average_emotions': {},
                'emotion_change_frequency': 0.0,
                'num_emotion_changes': 0,
                'processing_details': processing_details, # может содержать частичную информацию
                'error_message': str(e)
            }
        return {}

    def smooth_neutral_spikes(self, emotions: List[str]) -> List[str]:
        """
        Сглаживает резкие изменения эмоций, когда единичная нейтральная эмоция
        возникает между двумя одинаковыми ненейтральными эмоциями.

        Args:
            emotions: Список обнаруженных эмоций

        Returns:
            List[str]: Сглаженный список эмоций
        """
        if len(emotions) < 3:
            return emotions

        smoothed = emotions.copy() # Используем .copy() для создания поверхностной копии

        for i in range(1, len(smoothed) - 1):
            if (smoothed[i] == "neutral" and
                    smoothed[i - 1] == smoothed[i + 1] and
                    smoothed[i - 1] != "neutral"):
                smoothed[i] = smoothed[i - 1]

        return smoothed

    def detect_emotion(self, frame: np.ndarray) -> Optional[str]:
        """
        Определяет эмоцию на кадре.

        Args:
            frame: Кадр изображения в формате BGR (OpenCV)

        Returns:
            Optional[str]: Название эмоции или None, если не удалось определить
        """
        try:
            result = self.detector.detect_emotions(frame)
            if not result: # Лицо найдено, но эмоции не определены или лицо не найдено детектором FER
                return "neutral"
            emotions_data = result[0]['emotions']
            return max(emotions_data, key=emotions_data.get)
        except IndexError: # Может возникнуть, если result пустой, но прошел if not result (маловероятно с FER)
            print("Video emotion detection warning: No face detected or emotion data available in result.")
            return "neutral" # Возвращаем neutral если лицо не найдено или нет данных
        except Exception as e:
            # Логируем ошибку, но не прерываем весь анализ видео из-за одного кадра
            print(f"Video emotion detection error on a frame: {e}")
            return None # Возвращаем None, чтобы этот кадр был пропущен, а не засчитан как "unknown" или "neutral" по ошибке

    def calculate_emotion_changes(self, emotions: List[str]) -> int:
        """
        Подсчитывает количество изменений эмоций в последовательности.

        Args:
            emotions: Список эмоций

        Returns:
            int: Количество изменений эмоций
        """
        if not emotions or len(emotions) < 2:
            return 0

        changes = 0
        # current_emotion = emotions[0] # Не нужно, если начинаем цикл с 1
        for i in range(1, len(emotions)):
            if emotions[i] != emotions[i-1]:
                changes +=1
        return changes