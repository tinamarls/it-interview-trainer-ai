# import cv2
# import numpy as np
# import time
# import os
# from collections import Counter
#
# # Импортируем библиотеку deepface
# try:
#     from deepface import DeepFace
#     print("Successfully imported deepface.")
#     deepface_available = True
# except ImportError:
#     print("Error: deepface not found. Please install it ('pip install deepface').")
#     deepface_available = False
#
# # deepface имеет свои метки эмоций, которые могут немного отличаться или быть в другом порядке
# # По умолчанию deepface использует 7 эмоций: 'angry', 'fear', 'neutral', 'sad', 'disgust', 'happy', 'surprise'
# # Будем использовать метки, которые выдает deepface при анализе результатов.
#
# # --- 1. Класс для извлечения кадров ---
# class VideoProcessor:
#     def __init__(self, video_path: str, frame_skip: int = 0):
#         """
#         Инициализация обработчика видео для извлечения кадров.
#
#         Args:
#             video_path (str): Путь к видео файлу (.mp4, .webm).
#             frame_skip (int, optional): Количество кадров для пропуска между обрабатываемыми кадрами.
#                                         0 или 1 - обработка каждого кадра. Defaults to 0.
#         """
#         if not os.path.exists(video_path):
#             raise FileNotFoundError(f"Video file not found at {video_path}")
#
#         self.video_path = video_path
#         self.frame_skip = max(0, frame_skip) # Пропускать не меньше 0 кадров
#         self._cap = None # Объект cv2.VideoCapture
#         self._current_frame_index = 0 # Индекс кадра в исходном видео
#         self._processed_frame_index = 0 # Индекс кадра среди обработанных (с учетом пропуска)
#         self._video_fps = None
#         self._total_frames = None
#
#     def __enter__(self):
#         """Открывает видеофайл при входе в контекстный менеджер (with)."""
#         self._cap = cv2.VideoCapture(self.video_path)
#         if not self._cap.isOpened():
#             raise IOError(f"Could not open video file {self.video_path}")
#
#         self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
#         self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#         print(f"VideoProcessor: Opened {self.video_path}")
#         print(f"VideoProcessor: FPS={self._video_fps}, Total Frames={self._total_frames}, Frame Skip={self.frame_skip}")
#
#         self._current_frame_index = 0
#         self._processed_frame_index = 0
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         """Закрывает видеофайл при выходе из контекстного менеджера."""
#         if self._cap is not None:
#             self._cap.release()
#             print(f"VideoProcessor: Released {self.video_path}")
#
#     def __iter__(self):
#         """Делает объект итерируемым."""
#         return self
#
#     def __next__(self):
#         """Читает следующий кадр с учетом пропуска."""
#         if self._cap is None:
#             raise RuntimeError("VideoProcessor not initialized or video not opened. Use 'with VideoProcessor(...)'.")
#
#         while self._current_frame_index < self._total_frames:
#             ret, frame = self._cap.read()
#
#             if not ret:
#                 # Достигнут конец видео или ошибка чтения
#                 self._current_frame_index = self._total_frames # Ensure loop terminates
#                 raise StopIteration
#
#             current_frame_original_index = self._current_frame_index
#             self._current_frame_index += 1
#
#             # Пропускаем кадры, если задано frame_skip
#             if self.frame_skip > 0 and (current_frame_original_index) % (self.frame_skip + 1) != 0:
#                 continue
#
#             # Получаем timestamp обработанного кадра
#             # cap.get(cv2.CAP_PROP_POS_MSEC) может быть неточным, но это лучший доступный вариант
#             timestamp_sec = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
#
#             self._processed_frame_index += 1
#
#             # Возвращаем информацию о кадре и сам кадр
#             return {
#                 'frame_index_original': current_frame_original_index,
#                 'frame_index_processed': self._processed_frame_index -1, # Нумерация с 0 для обработанных
#                 'timestamp_sec': timestamp_sec,
#                 'frame': frame
#             }
#
#         # Завершение итерации
#         raise StopIteration
#
#     def get_fps(self) -> float:
#         """Возвращает FPS видео."""
#         return self._video_fps
#
#     def get_total_frames_original(self) -> int:
#         """Возвращает общее количество кадров в исходном видео."""
#         return self._total_frames
#
#     def get_total_frames_processed(self) -> int:
#         """Возвращает общее количество кадров, которые будут обработаны с учетом пропуска."""
#         if self._total_frames is None or self.frame_skip is None:
#             return 0
#         if self.frame_skip == 0:
#             return self._total_frames
#         return (self._total_frames + self.frame_skip) // (self.frame_skip + 1)
#
#
# # --- 2. Класс для анализа эмоций кадра ---
# class EmotionDetector:
#     def __init__(self):
#         """
#         Инициализация детектора эмоций с использованием deepface.
#         deepface загрузит модели при первом использовании функции analyze().
#         """
#         print("Initializing EmotionDetector using deepface...")
#         if not deepface_available:
#             print("deepface is not available. Emotion detection will not work.")
#
#         # Можно предварительно загрузить модель, но обычно analyze делает это сам
#         # try:
#         #     print("Pre-loading deepface emotion model...")
#         #     DeepFace.build_model('Emotion')
#         #     print("deepface emotion model pre-loaded.")
#         # except Exception as e:
#         #     print(f"Error pre-loading deepface model: {e}")
#
#
#     def process_frame(self, frame: np.array):
#         """
#         Обработка одного кадра для детекции лиц и классификации эмоций с помощью deepface.
#
#         Args:
#             frame (np.array): Изображение кадра (в формате BGR).
#
#         Returns:
#             list: Список словарей для каждого обнаруженного лица в кадре:
#                   {'bbox': (x1, y1, x2, y2),
#                    'emotion': 'DetectedEmotion', # Доминантная эмоция deepface (str)
#                    'confidence': 0.99, # Уверенность deepface в доминантной эмоции (float)
#                    'full_emotion_scores': {'angry': 0.1, ...} # Полный словарь уверенностей (dict)
#                    }
#                   Если deepface недоступен или лиц нет, возвращается пустой список.
#         """
#         if frame is None or not deepface_available:
#             return []
#
#         frame_results = []
#
#         try:
#             # deepface.analyze сам находит лица и анализирует эмоции
#             # img_path=frame: Передаем сам numpy массив кадра.
#             # actions=['emotion']: Запрашиваем только анализ эмоций.
#             # enforce_detection=False: Не выбрасывать исключение, если лицо не найдено, просто вернуть [].
#             # detector_backend='opencv': Можно выбрать детектор лиц, например, OpenCV (по умолчанию deepface выбирает сам).
#             detections = DeepFace.analyze(
#                 img_path=frame,
#                 actions=['emotion'],
#                 enforce_detection=False,
#                 # detector_backend='opencv'
#             )
#
#             # deepface.analyze для одного изображения/кадра возвращает список словарей,
#             # по одному словарю для каждого найденного лица.
#             # Если лиц не найдено и enforce_detection=False, возвращает пустой список [].
#
#             if isinstance(detections, list):
#                 for det in detections:
#                     # Извлекаем нужные данные из словаря deepface для этого лица
#                     emotion_scores = det.get('emotion', {})
#                     dominant_emotion = det.get('dominant_emotion', 'Unknown')
#                     region = det.get('region', {'x': 0, 'y': 0, 'w': 0, 'h': 0})
#
#                     # Рассчитываем confidence для доминантной эмоции (deepface часто возвращает ее как часть scores)
#                     # Если dominant_emotion='Unknown', confidence будет 0.0
#                     confidence = emotion_scores.get(dominant_emotion, 0.0)
#                     # deepface возвращает bbox в формате (x, y, w, h), переводим в (x1, y1, x2, y2)
#                     x1, y1, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
#                     bbox = (x1, y1, x1 + w, y1 + h)
#
#                     # Проверяем, что bbox корректен
#                     if w > 0 and h > 0:
#                         frame_results.append({
#                             'bbox': bbox,
#                             'emotion': dominant_emotion, # Строка, например 'happy'
#                             'confidence': confidence,
#                             'full_emotion_scores': emotion_scores # Полный словарь уверенностей
#                         })
#             # Если deepface вернул не список (что редко), пропускаем обработку
#
#         except Exception as e:
#             print(f"Error during deepface analyze call: {e}")
#             # В случае ошибки deepface для кадра, просто возвращаем пустой список результатов для этого кадра
#             frame_results = []
#
#         return frame_results
#
#     def analyze_results(self, all_frame_results: list, analysis_duration_sec: float = None, frame_timestamps: list = None):
#         """
#         Анализирует собранные результаты по всем кадрам для определения
#         преобладающей эмоции и частоты изменений.
#
#         Args:
#             all_frame_results (list): Список результатов обработки кадров.
#                                       Каждый элемент списка - это список словарей лиц из process_frame для одного кадра.
#                                       Пример: [[face1_f0, face2_f0], [face1_f1], ...]
#             analysis_duration_sec (float, optional): Общая длительность периода анализа в секундах.
#                                                     Если предоставлено, используется для расчета частоты. Defaults to None.
#             frame_timestamps (list, optional): Список timestamp'ов для каждого элемента в all_frame_results.
#                                                Если предоставлены, могут использоваться для более точного расчета длительности. Defaults to None.
#
#         Returns:
#             dict: Словарь с результатами анализа.
#                   'predominant_emotion': Преобладающая доминантная эмоция.
#                   'emotion_change_count': Общее количество изменений доминантной эмоции кадра.
#                   'total_frames_processed': Общее количество обработанных кадров.
#                   'total_detections': Общее количество обнаружений лиц во всех кадрах.
#                   'emotion_counts': Словарь с общим количеством каждого типа доминантной эмоции.
#                   'emotion_change_frequency_per_sec': Частота изменений эмоций в секундах.
#         """
#         if not all_frame_results:
#             return {
#                 'predominant_emotion': 'No frames processed',
#                 'emotion_change_count': 0,
#                 'total_frames_processed': 0,
#                 'total_detections': 0,
#                 'emotion_counts': {},
#                 'emotion_change_frequency_per_sec': 0.0
#             }
#
#         # deepface выдает доминантные эмоции маленькими буквами ('happy', 'sad')
#         # Получим список всех возможных доминантных эмоций из первой детекции, если она есть
#         deepface_labels = []
#         if all_frame_results and isinstance(all_frame_results[0], list) and all_frame_results[0]:
#             if 'full_emotion_scores' in all_frame_results[0][0] and isinstance(all_frame_results[0][0]['full_emotion_scores'], dict):
#                 deepface_labels = list(all_frame_results[0][0]['full_emotion_scores'].keys())
#         # Если не удалось получить из результатов, используем стандартные
#         if not deepface_labels:
#             deepface_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
#
#         # Создадим маппинг для вывода в регистре заголовка, если нужно
#         emotion_label_map = {label: label.capitalize() for label in deepface_labels}
#         # Добавим 'Unknown' на всякий случай
#         emotion_label_map['Unknown'] = 'Unknown'
#         emotion_label_map['No Faces'] = 'No Faces'
#
#
#         all_detected_emotions_flat = [] # Плоский список всех доминантных эмоций со всех лиц во всех кадрах
#         frame_dominant_emotions = []    # Список доминантных эмоций по кадрам (для расчета частоты изменений)
#
#         total_detections = 0
#
#         for frame_detections in all_frame_results:
#             # Список доминантных эмоций всех лиц в текущем кадре
#             emotions_in_frame = [det.get('emotion', 'Unknown') for det in frame_detections]
#             all_detected_emotions_flat.extend(emotions_in_frame)
#             total_detections += len(emotions_in_frame)
#
#             # Определяем доминирующую эмоцию в кадре для расчета частоты изменений
#             # Берем самую частую доминантную эмоцию среди РАСПОЗНАННЫХ deepface эмоций в этом кадре
#             recognized_frame_emotions = [e for e in emotions_in_frame if e in deepface_labels]
#
#             if recognized_frame_emotions:
#                 dominant_in_frame = Counter(recognized_frame_emotions).most_common(1)[0][0]
#                 frame_dominant_emotions.append(dominant_in_frame)
#             elif emotions_in_frame and "Unknown" in emotions_in_frame:
#                 frame_dominant_emotions.append("Unknown") # Если deepface вернул 'Unknown'
#             else: # Если лиц не было в кадре
#                 frame_dominant_emotions.append("No Faces")
#
#         # 1. Определение преобладающей эмоции за все видео
#         # Считаем частоту всех РАСПОЗНАННЫХ доминантных эмоций во всех кадрах
#         recognized_all_emotions = [e for e in all_detected_emotions_flat if e in deepface_labels]
#         emotion_counts = Counter(recognized_all_emotions)
#
#         predominant_emotion = "N/A"
#         if emotion_counts:
#             # Находим самую частую из распознанных
#             predominant_emotion = emotion_counts.most_common(1)[0][0]
#         elif total_detections > 0:
#             predominant_emotion = "No emotions recognized" # Если лица были, но ни одна эмоция не распознана
#         else:
#             predominant_emotion = "No faces detected"
#
#         # 2. Расчет частоты изменения эмоций
#         # Считаем количество переходов между разными доминантными РАСПОЗНАННЫМИ эмоциями кадра
#         emotion_changes = 0
#         previous_dominant_emotion = None
#         # Проходим по списку доминирующих эмоций по кадрам
#         for current_dominant_emotion in frame_dominant_emotions:
#             # Учитываем изменение только если текущая эмоция отличается от предыдущей
#             # И обе являются РАСПОЗНАННЫМИ deepface эмоциями (не 'No Faces' или 'Unknown')
#             if previous_dominant_emotion is not None and \
#                     current_dominant_emotion != previous_dominant_emotion and \
#                     current_dominant_emotion in deepface_labels and \
#                     previous_dominant_emotion in deepface_labels:
#                 emotion_changes += 1
#             previous_dominant_emotion = current_dominant_emotion
#
#         # Расчет частоты изменений в секундах
#         emotion_change_frequency_per_sec = 0.0
#         if analysis_duration_sec is not None and analysis_duration_sec > 0:
#             emotion_change_frequency_per_sec = emotion_changes / analysis_duration_sec
#         # Если длительность не передана, или переданы timestamp'ы
#         elif frame_timestamps is not None and len(frame_timestamps) > 1:
#             # Используем разницу между последним и первым timestamp'ами обработанных кадров
#             duration_from_timestamps = frame_timestamps[-1] - frame_timestamps[0]
#             if duration_from_timestamps > 0:
#                 emotion_change_frequency_per_sec = emotion_changes / duration_from_timestamps
#
#
#         # Форматируем названия эмоций для финального вывода
#         formatted_predominant = emotion_label_map.get(predominant_emotion, predominant_emotion)
#         formatted_emotion_counts = {emotion_label_map.get(k, k): v for k, v in emotion_counts.items()}
#
#
#         return {
#             'predominant_emotion': formatted_predominant,
#             'emotion_change_count': emotion_changes,
#             'total_frames_processed': len(all_frame_results), # Это число кадров, ПЕРЕДАННЫХ в analyze_results
#             'total_detections': total_detections,
#             'emotion_counts': formatted_emotion_counts,
#             'emotion_change_frequency_per_sec': emotion_change_frequency_per_sec
#         }
#
#
# # --- 3. Соединяющий класс для анализа видео ---
# class VideoAnalysis:
#     def __init__(self, video_path: str, frame_skip: int = 0):
#         """
#         Инициализация класса для анализа видео.
#
#         Args:
#             video_path (str): Путь к видео файлу (.mp4, .webm).
#             frame_skip (int, optional): Количество кадров для пропуска между обрабатываемыми кадрами.
#                                         0 или 1 - обработка каждого кадра. Defaults to 0.
#         """
#         print("Initializing VideoAnalysis...")
#         self.video_path = video_path
#         self.frame_skip = frame_skip
#         # Создаем экземпляры процессора видео и детектора эмоций
#         # VideoProcessor проверяет существование файла при инициализации или __enter__
#         # EmotionDetector проверяет доступность deepface
#         self.emotion_detector = EmotionDetector()
#         self._collected_frame_results = [] # Список для сбора результатов process_frame
#         self._processed_frame_timestamps = [] # Список для сбора timestamp'ов обработанных кадров
#
#
#     def perform_analysis(self):
#         """
#         Выполняет анализ видео: извлекает кадры, анализирует эмоции и собирает результаты.
#
#         Returns:
#             dict: Результаты анализа видео.
#                   Возвращает пустой словарь, если анализ не удался или deepface недоступен.
#         """
#         print(f"Performing analysis for video: {self.video_path}")
#
#         if not deepface_available:
#             print("Analysis cannot proceed: deepface is not available.")
#             return {}
#
#         self._collected_frame_results = [] # Сброс результатов предыдущего анализа
#         self._processed_frame_timestamps = []
#
#         # Используем VideoProcessor как контекстный менеджер для автоматического закрытия
#         try:
#             with VideoProcessor(self.video_path, self.frame_skip) as video_processor:
#                 total_frames_to_process = video_processor.get_total_frames_processed()
#                 print(f"VideoAnalysis: Expecting to process {total_frames_to_process} frames.")
#
#                 start_time_processing = time.time() # Время начала обработки
#
#                 # Итерируемся по кадрам, которые выдает VideoProcessor
#                 for frame_data in video_processor:
#                     # frame_data содержит: 'frame_index_original', 'frame_index_processed', 'timestamp_sec', 'frame'
#
#                     # Анализируем эмоции в текущем кадре с помощью EmotionDetector
#                     current_frame_emotion_results = self.emotion_detector.process_frame(frame_data['frame'])
#
#                     # Собираем результаты и timestamp
#                     self._collected_frame_results.append(current_frame_emotion_results)
#                     self._processed_frame_timestamps.append(frame_data['timestamp_sec'])
#
#                     # Опционально: визуализация (для отладки)
#                     # frame_display = frame_data['frame'].copy()
#                     # for res in current_frame_emotion_results:
#                     #      x1, y1, x2, y2 = res['bbox']
#                     #      emotion = res['emotion']
#                     #      confidence = res['confidence']
#                     #      label = f"{emotion}: {confidence:.2f}"
#                     #      cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     #      cv2.putText(frame_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                     # cv2.imshow('Video Analysis Frame', frame_display)
#                     # if cv2.waitKey(1) & 0xFF == ord('q'):
#                     #      break # Прерываем анализ
#
#                 end_time_processing = time.time() # Время окончания обработки
#                 actual_processing_duration_sec = end_time_processing - start_time_processing
#                 print(f"VideoAnalysis: Finished frame processing. Processed {len(self._collected_frame_results)} frames in {actual_processing_duration_sec:.2f} seconds.")
#                 # cv2.destroyAllWindows() # Закрываем окна визуализации, если они были открыты
#
#         except FileNotFoundError as e:
#             print(f"Error: {e}")
#             return {}
#         except IOError as e:
#             print(f"Error opening video: {e}")
#             return {}
#         except RuntimeError as e:
#             print(f"Runtime Error during video processing: {e}")
#             return {}
#         except Exception as e:
#             print(f"An unexpected error occurred during analysis: {e}")
#             return {}
#
#
#         # Анализируем собранные результаты всех обработанных кадров
#         # Передаем timestamp'ы для расчета частоты
#         final_analysis_results = self.emotion_detector.analyze_results(
#             self._collected_frame_results,
#             frame_timestamps=self._processed_frame_timestamps
#         )
#
#         # Добавляем общую информацию о процессе
#         final_analysis_results['video_path'] = self.video_path
#         final_analysis_results['frame_skip'] = self.frame_skip
#         final_analysis_results['total_frames_read_by_processor'] = video_processor.get_total_frames_original() # Извлечено из context manager
#         final_analysis_results['total_frames_processed_by_analyzer'] = len(self._collected_frame_results) # Передано в детектор
#         final_analysis_results['actual_processing_duration_sec'] = actual_processing_duration_sec # Время работы perform_analysis
#
#         return final_analysis_results
#
#
# # --- Пример использования всего пайплайна ---
# if __name__ == "__main__":
#     # --- УКАЖИТЕ ПУТЬ К ВАШЕМУ ВИДЕОФАЙЛУ ---
#     VIDEO_FILE_PATH = "/Users/kristina/IdeaProjects/it-train-diploma/machine-learning-service/soft_skills_analysis/temp/video/uploaded-11251824096247404542.mp4" # <-- ЗАМЕНИТЕ НА ПУТЬ К ВАШЕМУ ВИДЕО
#
#     if not os.path.exists(VIDEO_FILE_PATH):
#         print(f"FATAL ERROR: Video file not found at {VIDEO_FILE_PATH}")
#         print("Please set the correct path to your video file.")
#         exit()
#
#     if not deepface_available:
#         print("\nFATAL ERROR: deepface is not installed. Please install it to run this example.")
#         exit()
#
#     # Параметры анализа
#     # frame_skip = 0 # Обрабатывать каждый кадр (очень медленно!)
#     frame_skip = 9 # Обрабатывать каждый 10-й кадр (пропускать 9) - для ускорения и теста
#
#     # Создаем экземпляр основного класса VideoAnalysis
#     try:
#         analyzer = VideoAnalysis(
#             video_path=VIDEO_FILE_PATH,
#             frame_skip=frame_skip
#         )
#     except (FileNotFoundError, IOError, RuntimeError) as e:
#         print(f"Error initializing VideoAnalysis or opening video: {e}")
#         exit()
#     except Exception as e:
#         print(f"An unexpected error occurred during VideoAnalysis initialization: {e}")
#         exit()
#
#
#     # Запускаем анализ видео
#     analysis_results = analyzer.perform_analysis()
#
#     # Выводим результаты
#     if analysis_results: # Проверяем, что словарь результатов не пустой (т.е. анализ прошел)
#         print("\n--- Final Analysis Results ---")
#         print(f"Video Path: {analysis_results.get('video_path', 'N/A')}")
#         print(f"Frame Skip: {analysis_results.get('frame_skip', 'N/A')}")
#         print(f"Total Frames Read (Original Video): {analysis_results.get('total_frames_read_by_processor', 0)}")
#         print(f"Total Frames Processed for Analysis: {analysis_results.get('total_frames_processed_by_analyzer', 0)}")
#         print(f"Total Faces Detected & Classified: {analysis_results.get('total_detections', 0)}")
#         print(f"Actual Processing Duration: {analysis_results.get('actual_processing_duration_sec', 0.0):.2f} seconds")
#         print("-" * 20)
#         print(f"Predominant Emotion (most frequent dominant): {analysis_results.get('predominant_emotion', 'N/A')}")
#         print(f"Overall Emotion Counts (Dominant Emotions): {analysis_results.get('emotion_counts', {})}")
#         print(f"Total Emotion Changes (of dominant recognized emotion per frame): {analysis_results.get('emotion_change_count', 0)}")
#         print(f"Emotion Change Frequency (changes per second, based on processed frames timestamps): {analysis_results.get('emotion_change_frequency_per_sec', 0.0):.2f}")
#     else:
#         print("\nVideo analysis did not complete successfully.")