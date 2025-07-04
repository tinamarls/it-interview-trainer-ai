from soft_skills_analysis.video_analysis.utils.video_processor import VideoProcessor
from soft_skills_analysis.video_analysis.emotion_detector import VideoEmotionDetector
from soft_skills_analysis.utils.logger import logger
from typing import Dict
import time

class VideoAnalyzer:
    def __init__(self, frame_skip: int = 5):
        self.video_processor = VideoProcessor(frame_skip=frame_skip)
        self.emotion_detector = VideoEmotionDetector() # Этот объект отвечает за детектирование эмоций

    def analyze_video(self, video_path: str) -> Dict:
        """Основной метод анализа видео"""
        start_time = time.time()
        frames, timestamps = self.video_processor.extract_frames(video_path)
        if not frames:
            # В случае ошибки извлечения кадров, можно вернуть пустой или ошибочный результат
            # или выбросить исключение, как сейчас
            raise ValueError("Не удалось извлечь кадры из видео")

        # 1. Получение результатов анализа эмоций от детектора
        # Предполагается, что emotion_detector.analyze_video возвращает словарь,
        # содержащий 'dominant_emotion', 'emotion_distribution', 'emotion_sequence'
        emotion_result = self.emotion_detector.analyze_video(video_path)

        # 2. Расчет частоты смены эмоций
        # Предполагается, что emotion_detector.calculate_emotion_changes
        # использует 'emotion_sequence' из emotion_result и timestamps
        emotion_changes_frequency = self.emotion_detector.calculate_emotion_changes(
            emotion_result.get('emotion_sequence', []), # Используем .get для безопасности
            timestamps
        )

        logger.info(f"Анализ видео завершен за {time.time() - start_time:.2f} сек")
        logger.info(f"Результат анализа эмоций (детектор): {emotion_result}")
        logger.info(f"Частота смены эмоций (видео): {emotion_changes_frequency}")

        # Формирование итогового словаря
        video_analysis_data = {
            'dominant_emotion': emotion_result.get('dominant_emotion', 'unknown'),
            'emotion_distribution': emotion_result.get('emotion_distribution', {}), # Поле добавлено/подтверждено
            'average_emotions': emotion_result.get('emotion_distribution', {}),      # Поле добавлено (принимается равным emotion_distribution)
            'emotion_change_frequency': emotion_changes_frequency,                   # Поле добавлено/подтверждено
            'emotion_sequence': emotion_result.get('emotion_sequence', []),          # Дополнительно, может быть полезно
            'timestamps': timestamps,                                                # Дополнительно, для связки с emotion_sequence
            'processing_details': {
                'total_frames_processed': len(frames),
                'video_duration_seconds': timestamps[-1] if timestamps else 0,
                'frame_skip_setting': self.frame_skip
            }
        }
        return video_analysis_data

        # Старый комментарий, можно удалить или пересмотреть:
        # 1. вызываем emotion_detector
        # 2. вызываем open_face
        # 3. передаем в единый анализатор для вывода итогов