# import os
# import torch
# from typing import List, Dict, Any
# from collections import Counter, defaultdict
#
# from soft_skills_analysis.audio_analysis.utils.audio_processor import AudioProcessor
# from soft_skills_analysis.utils.logger import logger
#
# from aniemore.recognizers.voice import VoiceRecognizer
# from aniemore.models import HuggingFaceModel
#
# class AudioEmotionDetector:
#     def __init__(self):
#         self.audio_processor = AudioProcessor()
#         model = HuggingFaceModel.Voice.WavLM
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.recognizer = VoiceRecognizer(model, device)
#
#     def analyze_emotion(self, audio_path: str) -> Dict[str, Any]:
#         # 1. Разделить аудио на сегменты
#         segments = self.audio_processor.process_audio(audio_path)
#         logger.info(f"Получены сегменты: {segments}")
#
#         if not segments:
#             logger.warning("Не удалось извлечь ни одного сегмента речи из аудио — возвращается пустой результат.")
#             return {
#                 "average_emotions": {},
#                 "emotion_dynamics": {}
#             }
#
#         # 2. Анализ эмоций по сегментам
#         try:
#             predictions = self.recognizer.recognize(segments, return_single_label=False)
#         except Exception as e:
#             logger.error(f"Ошибка при анализе эмоций: {e}")
#             raise
#
#         # # 3. Удаление временных сегментов
#         # for seg in segments:
#         #     try:
#         #         os.remove(seg)
#         #     except Exception as e:
#         #         logger.warning(f"Не удалось удалить файл {seg}: {e}")
#
#         # 4. Агрегация результатов
#         logger.info(predictions)
#         average_emotions = self._aggregate(predictions)
#
#         # 5. Выбор анализа динамики
#         if len(predictions) <= 10:
#             dynamics = self._analyze_short_term_dynamics(predictions)
#         else:
#             dynamics = self._analyze_dynamics(predictions)
#
#         return {
#             "average_emotions": average_emotions,
#             "emotion_dynamics": dynamics
#         }
#
#     def _aggregate(self, segment_predictions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
#         """Усредняет уверенности по всем сегментам."""
#         emotion_sums = defaultdict(float)
#         total_segments = len(segment_predictions)
#
#         if total_segments == 0:
#             return {}
#
#         for emotions in segment_predictions.values():
#             for emotion, confidence in emotions.items():
#                 emotion_sums[emotion] += confidence
#
#         return {
#             emotion: round(total_conf / total_segments, 4)
#             for emotion, total_conf in emotion_sums.items()
#         }
#
#     def _analyze_short_term_dynamics(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
#         """Анализ смены эмоций для коротких аудио (до 10 сегментов)."""
#         sorted_segments = sorted(
#             predictions.items(),
#             key=lambda x: int(x[0].split('_')[-1].split('.')[0])
#         )
#
#         sequence = [max(e.items(), key=lambda x: x[1])[0] for _, e in sorted_segments]
#         transitions = [(prev, curr) for prev, curr in zip(sequence, sequence[1:]) if prev != curr]
#
#         return {
#             "changes_count": len(transitions),
#             "changes_ratio": round(len(transitions) / len(sequence), 2),
#             "was_stable": len(transitions) <= 2,
#             "main_transition": Counter(transitions).most_common(1)[0][0] if transitions else None
#         }
#
#     def _analyze_dynamics(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
#         """Более детальный анализ динамики для длинных аудио."""
#         sorted_segments = sorted(
#             predictions.items(),
#             key=lambda x: int(x[0].split('_')[-1].split('.')[0])
#         )
#
#         prev_dominant = None
#         dominant_changes = 0
#         transition_matrix = defaultdict(Counter)
#
#         for i, (_, emotions) in enumerate(sorted_segments):
#             current_dominant = max(emotions.items(), key=lambda x: x[1])[0]
#
#             if i > 0 and current_dominant != prev_dominant:
#                 dominant_changes += 1
#                 transition_matrix[prev_dominant][current_dominant] += 1
#
#             prev_dominant = current_dominant
#
#         total_segments = len(sorted_segments)
#         audio_duration_min = (total_segments * 3) / 60  # ~3 сек на сегмент
#         stability_index = 1 - (dominant_changes / total_segments)
#
#         return {
#             "transitions_total": total_segments - 1,
#             "transitions_per_minute": round(dominant_changes / audio_duration_min, 2),
#             "dominant_emotion_changes": dominant_changes,
#             "emotion_stability": round(stability_index, 2),
#             "transition_matrix": {
#                 src: dict(tgt_counts) for src, tgt_counts in transition_matrix.items()
#             }
#         }
#
#
#

import os
import torch
from typing import List, Dict, Any
from collections import Counter, defaultdict
from soft_skills_analysis.audio_analysis.utils.audio_processor import AudioProcessor
from soft_skills_analysis.utils.logger import logger
from aniemore.recognizers.voice import VoiceRecognizer
from aniemore.models import HuggingFaceModel
import librosa

class AudioEmotionDetector:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        model = HuggingFaceModel.Voice.WavLM
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.recognizer = VoiceRecognizer(model, device)

    def analyze_emotion(self, audio_path: str) -> Dict[str, Any]:
        # 1. Разделить аудио на сегменты
        segments = self.audio_processor.process_audio(audio_path)
        logger.info(f"Получены сегменты: {segments}")

        if not segments:
            logger.warning("Не удалось извлечь ни одного сегмента речи из аудио — возвращается пустой результат.")
            return {
                "average_emotions": {},
                "emotion_dynamics": {},
                "emotion_sequence": [],
                "segment_timestamps": []
            }

        # 2. Анализ эмоций по сегментам
        try:
            predictions = self.recognizer.recognize(segments, return_single_label=False)
        except Exception as e:
            logger.error(f"Ошибка при анализе эмоций: {e}")
            raise

        # 3. Создание последовательности эмоций и временных меток
        sorted_segments = sorted(
            predictions.items(),
            key=lambda x: int(x[0].split('_')[-1].split('.')[0])
        )
        emotion_sequence = [max(e.items(), key=lambda x: x[1])[0] for _, e in sorted_segments]

        # Генерация временных меток для сегментов
        segment_timestamps = []
        current_time = 0.0
        for seg_path in [s[0] for s in sorted_segments]:
            duration = librosa.get_duration(filename=seg_path)
            segment_timestamps.append(current_time)
            current_time += duration

        # 4. Агрегация результатов
        average_emotions = self._aggregate(predictions)

        # 5. Выбор анализа динамики
        if len(predictions) <= 10:
            dynamics = self._analyze_short_term_dynamics(predictions)
        else:
            dynamics = self._analyze_dynamics(predictions)

        # 6. Расчет частоты смены эмоций
        emotion_changes_per_minute = self._calculate_emotion_changes(emotion_sequence, segment_timestamps)

        return {
            "average_emotions": average_emotions,
            "emotion_dynamics": dynamics,
            "emotion_sequence": emotion_sequence,
            "segment_timestamps": segment_timestamps,
            "emotion_changes_per_minute": emotion_changes_per_minute
        }

    def _aggregate(self, segment_predictions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Усредняет уверенности по всем сегментам."""
        emotion_sums = defaultdict(float)
        total_segments = len(segment_predictions)

        if total_segments == 0:
            return {}

        for emotions in segment_predictions.values():
            for emotion, confidence in emotions.items():
                emotion_sums[emotion] += confidence

        return {
            emotion: round(total_conf / total_segments, 4)
            for emotion, total_conf in emotion_sums.items()
        }

    def _analyze_short_term_dynamics(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Анализ смены эмоций для коротких аудио (до 10 сегментов)."""
        sorted_segments = sorted(
            predictions.items(),
            key=lambda x: int(x[0].split('_')[-1].split('.')[0])
        )

        sequence = [max(e.items(), key=lambda x: x[1])[0] for _, e in sorted_segments]
        transitions = [(prev, curr) for prev, curr in zip(sequence, sequence[1:]) if prev != curr]

        return {
            "changes_count": len(transitions),
            "changes_ratio": round(len(transitions) / len(sequence), 2) if sequence else 0.0,
            "was_stable": len(transitions) <= 2,
            "main_transition": Counter(transitions).most_common(1)[0][0] if transitions else None
        }

    def _analyze_dynamics(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Более детальный анализ динамики для длинных аудио."""
        sorted_segments = sorted(
            predictions.items(),
            key=lambda x: int(x[0].split('_')[-1].split('.')[0])
        )

        prev_dominant = None
        dominant_changes = 0
        transition_matrix = defaultdict(Counter)

        for i, (_, emotions) in enumerate(sorted_segments):
            current_dominant = max(emotions.items(), key=lambda x: x[1])[0]

            if i > 0 and current_dominant != prev_dominant:
                dominant_changes += 1
                transition_matrix[prev_dominant][current_dominant] += 1

            prev_dominant = current_dominant

        total_segments = len(sorted_segments)
        audio_duration_min = (total_segments * 3) / 60  # ~3 сек на сегмент
        stability_index = 1 - (dominant_changes / total_segments) if total_segments > 0 else 0.0

        return {
            "transitions_total": total_segments - 1,
            "transitions_per_minute": round(dominant_changes / audio_duration_min, 2) if audio_duration_min > 0 else 0.0,
            "dominant_emotion_changes": dominant_changes,
            "emotion_stability": round(stability_index, 2),
            "transition_matrix": {
                src: dict(tgt_counts) for src, tgt_counts in transition_matrix.items()
            }
        }

    def _calculate_emotion_changes(self, emotion_sequence: List[str], timestamps: List[float]) -> float:
        """Рассчитывает частоту смены эмоций в минуту."""
        if len(emotion_sequence) < 2 or len(timestamps) < 2:
            return 0.0

        changes = sum(
            1 for i in range(1, len(emotion_sequence)) if emotion_sequence[i] != emotion_sequence[i - 1]
        )

        total_time = timestamps[-1] - timestamps[0] if timestamps else 0.0
        if total_time <= 0:
            return 0.0

        return (changes / total_time) * 60