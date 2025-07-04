from typing import Dict
import librosa

from soft_skills_analysis.utils.file_processor import FileProcessor
from soft_skills_analysis.utils.logger import logger

from soft_skills_analysis.audio_analysis.emotion_detector import AudioEmotionDetector
from soft_skills_analysis.audio_analysis.utils.audio_processor import AudioProcessor
from soft_skills_analysis.audio_analysis.speech_features import SpeechFeatureExtractor

class AudioAnalyzer:
    def __init__(self):
        self.emotion_detector = AudioEmotionDetector()
        self.audio_processor = AudioProcessor()
        self.speeach_feature_extractor = SpeechFeatureExtractor()
        self.file_processor = FileProcessor()

    def analyze(self, video_path: str) -> Dict:
        # Извлекаем аудио из видео
        audio_path = self.file_processor.extract_audio_from_video(video_path)

        # Эмоциональные критерии
        emotions = self.emotion_detector.analyze_emotion(audio_path)
        logger.info(emotions)

        # Лингвистические и паралингвистические критерии
        transcript = self.audio_processor.transcribe(audio_path)
        logger.info(transcript)
        speech_features = self.speeach_feature_extractor.extract_features(audio_path, transcript, self.audio_processor.get_audio_duration(audio_path))
        logger.info(speech_features)

        return emotions