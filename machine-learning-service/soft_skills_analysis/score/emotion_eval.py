import os
import subprocess  # Добавлено для FFmpeg
import tempfile  # Для создания временных файлов/директорий, если потребуется
import time
from pathlib import Path

import cv2
import dlib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fer import FER
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.signal import savgol_filter
from scipy.spatial import distance as dist

import google.generativeai as genai  # Правильный импорт

from soft_skills_analysis.video_analysis.open_face_detector import OpenFaceDetector
from soft_skills_analysis.video_analysis.utils.video_processor import VideoProcessor

try:
    from soft_skills_analysis.score.new_eval.audio_eval import AudioEmotionAnalyzer
except ImportError:
    print(
        "Предупреждение: не удалось импортировать AudioEmotionAnalyzer. Функционал анализа аудиоэмоций будет недоступен.")
    AudioEmotionAnalyzer = None

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from faster_whisper import WhisperModel

print("Используется faster-whisper.")
# faster-whisper требует указания compute_type: "float16" (GPU), "int8" (CPU) и т.д.
# CPU_COMPUTE_TYPE = "int8" # для CPU
CPU_COMPUTE_TYPE = "float32"  # Более универсальный для CPU, но медленнее int8
GPU_COMPUTE_TYPE = "float16"  # для GPU
# Установите нужный тип ниже
COMPUTE_TYPE = CPU_COMPUTE_TYPE  # или GPU_COMPUTE_TYPE если есть GPU

import logging

logging.basicConfig(level=logging.INFO)  # Можно заменить на DEBUG
logger = logging.getLogger(__name__)

class EmotionEvaluator:
    """
    Класс для анализа эмоций, речи и визуальных сигналов
    в видеозаписи интервью с использованием Whisper, FER и dlib.
    """

    def __init__(self,
                 whisper_model_size="medium",  # "tiny", "base", "small", "medium", "large"
                 dlib_predictor_path="shape_predictor_68_face_landmarks.dat",
                 silence_thresh_db=-40,
                 min_silence_len_ms=500,
                 device="cpu",
                 ):
        self.config = {
            'silence_thresh': silence_thresh_db,
            'min_silence_len': min_silence_len_ms,
        }
        self.video_processor = VideoProcessor()
        self.open_face_detector = OpenFaceDetector()
        # ... (загрузка моделей Whisper, FER) ...
        print("Загрузка модели Whisper...")
        try:
            compute_type_to_use = GPU_COMPUTE_TYPE if device == "cuda" else CPU_COMPUTE_TYPE
            print(f"Загрузка faster-whisper с compute_type='{compute_type_to_use}' на {device}")
            self.whisper_model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type_to_use)
            print("Модель Whisper загружена.")
        except Exception as e:
            print(f"Ошибка загрузки модели Whisper: {e}")
            raise

        # --- Загрузка FER (Emotion Detection) ---
        print("Загрузка детектора эмоций (FER)...")
        try:
            self.emotion_detector = FER(mtcnn=True)
            print("Детектор эмоций FER загружен.")
        except Exception as e:
            print(f"Ошибка загрузки FER: {e}")
            raise

        # --- Загрузка Dlib (Face Landmarks) ---
        print("Загрузка детектора лицевых ориентиров (dlib)...")
        if not os.path.exists(dlib_predictor_path):
            raise FileNotFoundError(f"Файл предиктора dlib не найден: {dlib_predictor_path}")
        try:
            self.face_detector_dlib = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(dlib_predictor_path)
            # Глаза для моргания
            (self.lStart, self.lEnd) = (42, 48)
            (self.rStart, self.rEnd) = (36, 42)
            # Ключевые точки для анализа направления головы/взгляда
            self.nose_tip_idx = 30
            self.chin_idx = 8
            self.left_eye_corner_idx = 36
            self.right_eye_corner_idx = 45
            self.left_brow_idx_outer = 17
            self.right_brow_idx_outer = 26
            print("Детектор dlib загружен.")
        except Exception as e:
            print(f"Ошибка загрузки dlib: {e}")
            raise
        print("EmotionEvaluator инициализирован.")

    def _extract_audio(self, video_path, audio_path):
        print("Извлечение аудио...")
        try:
            with VideoFileClip(video_path) as video_clip:
                video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
            print(f"Аудио сохранено в {audio_path}")
            return True
        except Exception as e:
            print(f"Ошибка извлечения аудио: {e}")
            return False

    def _transcribe_audio_whisper(self, audio_path):
        print("Распознавание речи (Whisper)...")
        try:
            segments_gen, info = self.whisper_model.transcribe(audio_path, language='ru', word_timestamps=False)
            segments = list(segments_gen)
            full_text = " ".join([s.text.strip() for s in segments])
            segments_list = [{'start': s.start, 'end': s.end, 'text': s.text} for s in segments]
            print(f"Язык распознан как: {info.language} с вероятностью {info.language_probability:.2f}")
            print("Текст распознан.")
            return full_text, segments_list
        except Exception as e:
            print(f"Ошибка распознавания речи Whisper: {e}")
            return None, []

    def _calculate_speech_rate(self, segments):
        total_duration = 0
        total_words = 0
        if not segments: return 0, []
        for segment in segments:
            start_time, end_time, text = segment['start'], segment['end'], segment['text']
            words = text.strip().split()
            num_words = len(words)
            if num_words > 0:
                duration = end_time - start_time
                if duration > 0.1:
                    total_duration += duration
                    total_words += num_words
        average_wpm = (total_words / total_duration) * 60 if total_duration > 0 else 0
        return average_wpm, []  # word_timestamps пока не реализованы

    def _detect_pauses(self, audio_path):
        print("Детекция пауз...")
        try:
            audio = AudioSegment.from_wav(audio_path)
            nonsilent_ranges = detect_nonsilent(
                audio, min_silence_len=self.config['min_silence_len'],
                silence_thresh=self.config['silence_thresh'],
                seek_step=1  # ms, для более точного определения границ
            )
            pauses, last_end_ms = [], 0
            for start_ms, end_ms in nonsilent_ranges:
                if start_ms > last_end_ms:
                    pause_duration_ms = start_ms - last_end_ms
                    # Проверяем, что пауза не слишком короткая (уже учтено в min_silence_len, но можно добавить доп. порог)
                    if pause_duration_ms >= self.config['min_silence_len']:
                        pauses.append({'start': last_end_ms / 1000.0, 'duration': pause_duration_ms / 1000.0})
                last_end_ms = end_ms

            # Проверка паузы в конце аудио
            if len(audio) > last_end_ms:
                pause_duration_ms = len(audio) - last_end_ms
                if pause_duration_ms >= self.config['min_silence_len']:
                    pauses.append({'start': last_end_ms / 1000.0, 'duration': pause_duration_ms / 1000.0})

            total_pause_time = sum(p['duration'] for p in pauses)
            print(f"Найдено пауз: {len(pauses)}, общая длительность: {total_pause_time:.2f} сек")
            return pauses, total_pause_time, len(pauses)
        except Exception as e:
            print(f"Ошибка детекции пауз: {e}")
            return [], 0, 0

    def compute_ttr(self, transcript: str) -> float:
        if not transcript: return 0.0
        tokens = transcript.lower().split()
        if not tokens: return 0.0
        types = set(tokens)
        return len(types) / len(tokens)

    def hesitation_coefficient(self, audio_path: str) -> float:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            print(f"Аудиофайл {audio_path} не существует или пуст. Пропуск расчета коэффициента хезитации.")
            return 0.0
        try:
            y, sr = librosa.load(audio_path, sr=16000)  # Загружаем аудио
            total_duration = len(y) / sr
            if total_duration == 0: return 0.0

            # Определение нетихих интервалов (речи)
            # top_db - порог в dB ниже максимальной громкости, который считается тишиной
            intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
            speech_duration = sum((end - start) / sr for start, end in intervals)
            pause_duration = total_duration - speech_duration

            return pause_duration / total_duration if total_duration > 0 else 0.0
        except Exception as e:
            print(f"Ошибка расчета коэффициента хезитации: {str(e)}")
            return 0.0

    def _eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C) if C != 0 else 0.0  # Избегаем деления на ноль

    def _analyze_video_frames(self, video_path):
        print("Анализ видео кадров (включая анализ направления взгляда)...")
        cap = cv2.VideoCapture(video_path)
        # ... (проверки cap.isOpened(), fps, frame_count, duration_sec как раньше) ...
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть видеофайл {video_path}")
            return pd.DataFrame(), {}, 0, 0, 0, 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps is None or fps == 0 or frame_count is None or frame_count == 0:
            # ... (fallback логика для duration_sec как раньше) ...
            try:
                clip = VideoFileClip(video_path)
                duration_sec_moviepy = clip.duration
                clip.close()
                duration_sec = duration_sec_moviepy
                if (fps is None or fps == 0) and frame_count > 0:
                    fps = 25  # Предположение
                elif frame_count == 0:
                    duration_sec = 0
            except Exception:
                duration_sec = 0
            if duration_sec == 0: return pd.DataFrame(), {}, 0, 0, 0, 0, 0, 0, 0, 0, 0
        else:
            duration_sec = frame_count / fps
        print(f"Видео: ~{frame_count} кадров, FPS: {fps:.2f}, Длительность: {duration_sec:.2f} сек")

        results = []
        frame_num = 0
        frames_with_face_and_emotion = 0  # Кадры, где FER нашел лицо и эмоцию
        frames_with_face_dlib = 0  # Кадры, где dlib нашел лицо (для % отведения взгляда)

        processing_interval = max(1, int(fps / 5)) if fps > 5 else 1
        print(f"Анализ каждого {processing_interval}-го кадра.")
        actual_frames_processed_count = 0
        time_per_processed_segment = duration_sec / (
                frame_count / processing_interval) if frame_count > 0 and processing_interval > 0 and duration_sec > 0 else (
            1 / fps if fps > 0 else 0.2)  # примерная длительность сегмента

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            timestamp = frame_num / fps if fps > 0 else (actual_frames_processed_count * time_per_processed_segment)

            if frame_num % processing_interval == 0:
                actual_frames_processed_count += 1
                current_emotion = 'no_face'  # По умолчанию, если лицо не найдено
                current_emotion_scores = {}
                current_ear = -1.0  # Используем float для EAR
                face_detected_by_fer = False

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_dlib = self.face_detector_dlib(gray, 0)

                if faces_dlib:
                    frames_with_face_dlib += 1
                    shape = self.landmark_predictor(gray, faces_dlib[0])
                    landmarks_np = np.array([[p.x, p.y] for p in shape.parts()])

                    # Анализ морганий (как раньше)
                    leftEye = landmarks_np[self.lStart:self.lEnd]
                    rightEye = landmarks_np[self.rStart:self.rEnd]
                    leftEAR = self._eye_aspect_ratio(leftEye)
                    rightEAR = self._eye_aspect_ratio(rightEye)
                    current_ear = (leftEAR + rightEAR) / 2.0

                # Детекция эмоций (как раньше)
                try:
                    emotion_result = self.emotion_detector.detect_emotions(frame)
                    if emotion_result:
                        face_detected_by_fer = True
                        frames_with_face_and_emotion += 1
                        emotions = emotion_result[0]["emotions"]
                        current_emotion = max(emotions, key=emotions.get)
                        current_emotion_scores = emotions
                except Exception as e_fer:  # Ошибка FER
                    # print(f"Ошибка FER на кадре {frame_num}: {e_fer}")
                    current_emotion = 'error_fer'  # Отдельная метка для ошибки FER

                results.append({
                    'timestamp': timestamp, 'frame': frame_num,
                    'dominant_emotion': current_emotion, 'emotion_scores': current_emotion_scores,
                    'ear': current_ear,
                })

            frame_num += 1
            if frame_num % (int(fps * 10 if fps > 0 else 250)) == 0:
                print(
                    f"Обработано {frame_num}/{frame_count if frame_count else 'N/A'} кадров...")

        cap.release()
        print(f"Анализ видео завершен. Всего логически обработано кадров: {actual_frames_processed_count}")

        if not results:  # Если results пуст, возвращаем нули для всех метрик
            return pd.DataFrame(), [], 0, 0, 0, 0, 0, 0, 0, 0, 0

        df = pd.DataFrame(results)

        # Расчет метрик отведения взгляда
        # Общее время, когда dlib находил лицо
        duration_sec_with_face_dlib = frames_with_face_dlib * time_per_processed_segment

        print(
            f"Общее кол-во кадров с лицом (dlib): {frames_with_face_dlib}, Длительность: {duration_sec_with_face_dlib:.2f} сек")

        return (df, df.to_dict(orient='records'),
                duration_sec)

    def _plot_emotions(self, df, output_path=None, smooth_window: int = 9, polyorder: int = 2):
        if df.empty or 'timestamp' not in df.columns or 'dominant_emotion' not in df.columns:
            print("Нет данных для построения графика эмоций.")
            return None

        emotion_rus = {
            'neutral': 'Нейтральная', 'angry': 'Злость', 'fear': 'Страх',
            'surprise': 'Удивление', 'happy': 'Позитив', 'sad': 'Грусть',
            'no_face': 'Лицо не найдено', 'error': 'Ошибка детекции'
        }
        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 7))  # Немного увеличил высоту

        # Создаем копию для плота, исключаем только 'error'
        df_plot = df[df['dominant_emotion'] != 'error'].copy()
        if df_plot.empty:
            print("Нет валидных данных об эмоциях для графика (после исключения 'error').")
            return None

        # Определяем порядок эмоций для графика
        # Сначала "позитивные", потом "нейтральные", потом "негативные", потом "технические"
        ordered_emotions_preferred = ['no_face', 'angry', 'fear', 'sad', 'neutral', 'surprise', 'happy']
        present_emotions = df_plot['dominant_emotion'].unique()
        plot_ordered_emotions = [e for e in ordered_emotions_preferred if e in present_emotions]
        plot_ordered_emotions.extend([e for e in present_emotions if e not in plot_ordered_emotions])

        emotion_map = {name: i for i, name in enumerate(plot_ordered_emotions)}
        df_plot['emotion_code'] = df_plot['dominant_emotion'].map(emotion_map)

        # Проверка на случай, если после map остались NaN (не должно быть, если plot_ordered_emotions корректен)
        if df_plot['emotion_code'].isnull().any():
            print("Предупреждение: Обнаружены NaN в 'emotion_code' перед построением графика.")
            df_plot.dropna(subset=['emotion_code'], inplace=True)  # Удаляем строки, где код эмоции не определился
            if df_plot.empty:
                print("Нет данных для графика после удаления строк с неопределенным кодом эмоции.")
                return None

        x = df_plot['timestamp'].values
        y = df_plot['emotion_code'].values

        # Сглаживание, если достаточно данных и окно нечетное
        # Убедимся, что smooth_window нечетное
        if smooth_window % 2 == 0: smooth_window += 1

        if len(y) >= smooth_window and len(y) > polyorder and polyorder > 0:
            try:
                y_smooth = savgol_filter(y, smooth_window, polyorder)
            except ValueError as e_savgol:  # Например, если window_length > x.shape[-1]
                print(f"Ошибка сглаживания Savitzky-Golay: {e_savgol}. Используются сырые данные.")
                y_smooth = y
        else:
            print(
                "Недостаточно данных для сглаживания или некорректные параметры, используются сырые данные для графика эмоций.")
            y_smooth = y

        plt.plot(x, y_smooth, linewidth=2, color='dodgerblue', label="Тренд эмоции")

        yticks_idx = list(range(len(plot_ordered_emotions)))
        # Убедимся, что yticks_labels соответствует plot_ordered_emotions, которые реально есть на графике
        # Если какие-то эмоции были отфильтрованы (например, из-за NaN в emotion_code), их не должно быть в метках
        current_emotion_codes_on_plot = sorted(df_plot['emotion_code'].unique())
        # Обновляем yticks_idx и yticks_labels на основе фактически присутствующих кодов
        final_yticks_idx = [i for i in yticks_idx if i in current_emotion_codes_on_plot]
        final_yticks_labels = [emotion_rus.get(plot_ordered_emotions[i], plot_ordered_emotions[i]) for i in
                               final_yticks_idx]

        if final_yticks_idx:  # Только если есть что отображать
            plt.yticks(final_yticks_idx, final_yticks_labels)

        plt.xlabel("Время (секунды)")
        plt.ylabel("Доминирующая эмоция")
        plt.title("Изменение доминирующей эмоции по времени (сглажено)")
        plt.grid(axis='y', linestyle='--')  # Только горизонтальные линии сетки
        plt.legend(loc='upper right')
        plt.tight_layout()

        if output_path:
            try:
                plt.savefig(output_path)
                print(f"График сохранен: {output_path}")
                plt.close()
                return output_path
            except Exception as e_save:
                print(f"Не удалось сохранить график: {e_save}")
                plt.show()  # Показать, если не удалось сохранить
                plt.close()
                return "plt_shown_due_to_save_error"
        else:
            plt.show()
            plt.close()  # Закрываем фигуру после показа
            return "plt_shown"

    def _calculate_stats(self, df_video, total_blinks, blink_rate, pauses, total_pause_time, num_pauses, avg_wpm,
                         transcript, duration_sec,
                         total_smile_duration_seconds, smile_percentage, ttr, hes_coeff,
                         total_aversion_duration_seconds, aversion_percentage,
                         aversion_frequency_per_minute, gaze_shift_frequency_per_minute, total_gaze_aversions):
        stats = {
            "video_duration_sec": duration_sec,
            "emotion_distribution": {},
            "avg_wpm": avg_wpm,
            "num_pauses": num_pauses,
            "total_pause_time_sec": total_pause_time,
            "avg_pause_duration_sec": 0,
            "total_blinks": total_blinks,
            "blink_rate_per_minute": blink_rate,
            "transcript_length_chars": len(transcript) if transcript else 0,
            "transcript_snippet": transcript,
            "total_smile_duration_seconds": total_smile_duration_seconds,
            "smile_percentage_of_face_time": smile_percentage,
            "ttr_lexical_diversity": ttr,
            "hesitation_coefficient": hes_coeff,
            # Новые метрики взгляда
            "total_gaze_aversion_duration_seconds": total_aversion_duration_seconds,
            "gaze_aversion_percentage_of_face_time": aversion_percentage,  # % от времени когда dlib нашел лицо
            "gaze_aversion_frequency_per_minute": aversion_frequency_per_minute,
            "gaze_shift_frequency_per_minute": gaze_shift_frequency_per_minute,
            "total_gaze_aversions": total_gaze_aversions
        }
        # ... (расчет emotion_distribution и avg_pause_duration_sec как раньше) ...
        if not df_video.empty and 'dominant_emotion' in df_video.columns:
            valid_emotions_df = df_video[~df_video['dominant_emotion'].isin(['error_fer', 'no_face'])]
            if not valid_emotions_df.empty:
                emotion_counts = valid_emotions_df['dominant_emotion'].value_counts()
                emotion_total = emotion_counts.sum()
                if emotion_total > 0:
                    stats["emotion_distribution"] = (emotion_counts / emotion_total * 100).to_dict()
        if num_pauses > 0 and total_pause_time > 0:
            stats["avg_pause_duration_sec"] = total_pause_time / num_pauses

        print("\n--- Рассчитанная Статистика ---")
        # ... (вывод статистики как раньше, он уже итерирует по словарю)
        for key, value in stats.items():
            if key == "emotion_distribution":
                print(f"{key.replace('_', ' ').capitalize()}:")
                if value:
                    for emo, perc in value.items(): print(f"  - {emo.capitalize()}: {perc:.1f}%")
                else:
                    print("  (Нет данных)")
            elif isinstance(value, float):
                print(f"{key.replace('_', ' ').capitalize()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').capitalize()}: {value}")
        return stats

    def _generate_advice(self, stats):
        prompt = f"""
            Ты - опытный HR-специалист, проводящий анализ soft skills кандидата по видеоинтервью. 
            Проанализируй следующие метрики и дай развернутые рекомендации:
    
            1. **Визуальные показатели**:
            - Доля отведенного взгляда: {stats.get('gaze_aversion_percentage_of_face_time', 0):.1f}%
            - Доля времени с улыбкой: {stats.get('smile_percentage_of_face_time', 0):.1f}%
            - Частота моргания: {stats.get('blink_rate_per_minute', 0):.1f}/мин
    
            2. **Речевые характеристики**:
            - Темп речи: {stats.get('avg_wpm', 0):.1f} слов/мин
            - Доля пауз в речи: {stats.get('total_pause_time_sec', 0):.1f} сек ({stats.get('num_pauses', 0)} пауз)
            - Лексическое разнообразие (TTR): {stats.get('ttr_lexical_diversity', 0):.2f}
            - Коэффициент хезитации: {stats.get('hesitation_coefficient', 0)*100:.1f}%
    
            3. **Эмоциональный профиль**:
            {self._format_emotion_distribution(stats.get('emotion_distribution', {}))}
    
            Проанализируй эти данные и дай:
            1. Оценку soft skills по шкале от 1 до 10
            2. 3 сильных стороны кандидата
            3. 3 зоны для развития
            4. Конкретные рекомендации по улучшению
            5. Общий вывод о соответствии коммуникативных навыков профессиональным требованиям
    
            Ответ оформляй в виде структурированного текста с маркированными списками. Кратко
        """

        try:
            logger.info("Настройка Gemini API")
            genai.configure(api_key='AIzaSyBMYbQhNa_OmIpEvRwt78UeiW9aI-QyRQU')

            model = genai.GenerativeModel(model_name="gemini-1.5-flash")

            retries = 0
            while retries < 3:
                try:
                    logger.info(f"Попытка {retries + 1} генерации ответа")
                    response = model.generate_content(prompt)
                    logger.info("Успешный ответ от модели")
                    return response.text
                except Exception as error:
                    logger.warning(f"Ошибка при вызове модели: {error}")
                    if '429' in str(error):
                        sleep_time = 5 * (retries + 1)
                        logger.info(f"Повтор через {sleep_time} сек...")
                        time.sleep(sleep_time)
                        retries += 1
                    else:
                        break

            logger.error("Превышено количество попыток вызова модели Gemini")
            return "Не удалось получить рекомендации от Gemini API"

        except Exception as e:
            logger.exception("Ошибка при генерации рекомендаций (внешний блок try)")
            return "Ошибка при генерации рекомендаций"

    def _format_emotion_distribution(self, emotion_dist):
        """Форматирование распределения эмоций для промпта"""
        if not emotion_dist:
            return "- Эмоции: данные недоступны"

        emotions_ru = {
            'neutral': 'Нейтральная',
            'happy': 'Радость',
            'surprise': 'Удивление',
            'angry': 'Злость',
            'sad': 'Грусть',
            'fear': 'Страх'
        }

        return "\n".join(
            f"- {emotions_ru.get(emo, emo)}: {perc:.1f}%"
            for emo, perc in emotion_dist.items()
        )

    def _fix_video_metadata_with_ffmpeg(self, input_path: str) -> str:
        temp_dir = tempfile.gettempdir()
        input_path_obj = Path(input_path)
        output_path = str(Path(temp_dir) / f"{input_path_obj.stem}_fixed{input_path_obj.suffix}")
        try:
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(input_path),
                "-c:v", "libvpx-vp9", "-crf", "28", "-b:v", "0",
                # CRF 28 - хорошее качество для VP9, не слишком большое
                "-deadline", "realtime", "-cpu-used", "4",  # Параметры для ускорения VP9
                "-c:a", "libopus", "-b:a", "96k",  # Opus с меньшим битрейтом для речи
                "-movflags", "faststart",
                "-y", output_path,
            ]
            print(f"Попытка стандартизировать видео FFmpeg: {input_path} -> {output_path}")
            process = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8',
                                     errors='ignore')

            if process.returncode != 0:
                print(f"Ошибка FFmpeg (код {process.returncode}) при обработке {input_path}:")
                print(f"FFmpeg stdout: {process.stdout.strip()}")
                print(f"FFmpeg stderr: {process.stderr.strip()}")
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                return str(input_path)  # Возвращаем оригинал

            print(f"FFmpeg обработка успешна. Файл сохранен: {output_path}")
            return output_path
        except FileNotFoundError:
            print("FFmpeg не найден. Пропуск этапа стандартизации видео. Используется оригинальный файл.")
            return str(input_path)
        except Exception as e_gen:
            print(f"Непредвиденная ошибка при вызове FFmpeg: {e_gen}")
            return str(input_path)

    def analyze_interview(self, video_path, plot_output_path=None):
        original_video_path_str = str(video_path)
        if not os.path.exists(original_video_path_str):
            return {"stats": self._get_default_stats_on_error("Видеофайл не найден."), "timeline_df": [],
                    "advice": ["Видеофайл не найден."], "plot_path": None, "audio_plot_path": None, "audio_stats": {}}

        start_time_analysis = time.time()
        video_path_for_analysis = self._fix_video_metadata_with_ffmpeg(original_video_path_str)
        temp_audio_path = ""  # Инициализация
        try:
            temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio_path = temp_audio_file.name
            temp_audio_file.close()

            analysis_results = {"stats": {}, "timeline_df": [], "advice": [], "plot_path": None,
                                "audio_plot_path": None, "audio_stats": {}}
            processed_ffmpeg_file_to_delete = None
            if video_path_for_analysis != original_video_path_str and os.path.exists(video_path_for_analysis):
                processed_ffmpeg_file_to_delete = video_path_for_analysis

            audio_extraction_successful = self._extract_audio(video_path_for_analysis, temp_audio_path)
            if not audio_extraction_successful:
                audio_extraction_successful = self._extract_audio(original_video_path_str, temp_audio_path)
            if not audio_extraction_successful:
                # ... (обработка ошибки извлечения аудио как раньше)
                print("Критическая ошибка: Не удалось извлечь аудио.")
                # Попытка проанализировать только видео без аудио метрик
                (df_video_err, _, _, _, duration_sec_err, _, _, _, _, _, _) = self._analyze_video_frames(
                    video_path_for_analysis)
                analysis_results["stats"] = self._get_default_stats_on_error("Ошибка извлечения аудио.",
                                                                             video_duration_sec=duration_sec_err if 'duration_sec_err' in locals() else 0)
                analysis_results["advice"].append(
                    "Критическая ошибка: Не удалось извлечь аудио. Аудиометрики не будут рассчитаны.")
                if plot_output_path and not df_video_err.empty:
                    analysis_results["plot_path"] = self._plot_emotions(df_video_err, output_path=plot_output_path)
                return analysis_results

            transcript, segments = self._transcribe_audio_whisper(temp_audio_path)
            if transcript is None: transcript, segments = "", []

            # Аудио анализ (AudioEmotionAnalyzer) - как раньше
            audio_plot_name = "audio_emotion_plot.png"
            if plot_output_path:
                base, ext = os.path.splitext(plot_output_path)
                audio_plot_name = base + "_audio" + ext
            try:
                print(f"Запуск анализа аудио эмоций с AudioEmotionAnalyzer (вывод в {audio_plot_name})...")
                audio_analyzer = AudioEmotionAnalyzer(original_audio_path=temp_audio_path, segments=segments,
                                                      plot_output_path=audio_plot_name, transcript=transcript)
                audio_anal_result_df = audio_analyzer.analyze_audio()
                if audio_anal_result_df is not None and not audio_anal_result_df.empty:
                    print("\nРезультаты аудио анализа (первые 5 сегментов):")
                    print(audio_anal_result_df.head())
                    analysis_results["audio_plot_path"] = audio_analyzer.plot_smoothed_dominant_emotion_curve()
                    analysis_results["audio_stats"] = audio_analyzer.get_statistics()
                else:
                    print("\nАудио анализ (AudioEmotionAnalyzer) не дал результатов.")
            except Exception as e_audio_analyzer:
                print(f"Ошибка во время аудио анализа (AudioEmotionAnalyzer): {e_audio_analyzer}")

            avg_wpm, _ = self._calculate_speech_rate(segments)
            pauses, total_pause_time, num_pauses = self._detect_pauses(temp_audio_path)
            ttr = self.compute_ttr(transcript)
            hes_coeff = self.hesitation_coefficient(temp_audio_path)

            # Обновленный вызов _analyze_video_frames
            (df_video, df_video_dict,
             video_duration_sec) = self._analyze_video_frames(
                video_path_for_analysis)
            # TODO вернуть
            # analysis_results["timeline_df"] = df_video_dict
            df_for_open_face = self.video_processor.get_open_face_csv(video_path_for_analysis)
            openFace_blink_frequency = self.open_face_detector.calculate_blink_frequency(df_for_open_face)
            openFace_smile_metrics = self.open_face_detector.calculate_smile_metrics(df_for_open_face)
            openFace_gaze_aversion = self.open_face_detector.calculate_gaze_aversion(df_for_open_face)
            total_blinks = openFace_blink_frequency["total_blinks"]
            blink_rate = openFace_blink_frequency["blink_frequency_bpm"]
            total_smile_duration_seconds = openFace_smile_metrics["total_smile_duration_seconds"]
            smile_percentage = openFace_smile_metrics["smile_percentage"]
            total_aversion_duration_seconds = openFace_gaze_aversion["total_aversion_duration_seconds"]
            aversion_percentage = openFace_gaze_aversion["aversion_percentage"]
            aversion_frequency_per_minute = openFace_gaze_aversion["aversion_frequency_per_minute"]
            gaze_shift_frequency_per_minute = openFace_gaze_aversion["gaze_shift_frequency_per_minute"]
            total_gaze_aversions = openFace_gaze_aversion["total_gaze_aversions"]

            # Обновленный вызов _calculate_stats
            stats = self._calculate_stats(
                df_video, total_blinks, blink_rate, pauses,
                total_pause_time, num_pauses, avg_wpm, transcript, video_duration_sec,
                total_smile_duration_seconds, smile_percentage, ttr, hes_coeff,
                total_aversion_duration_seconds, aversion_percentage,
                aversion_frequency_per_minute, gaze_shift_frequency_per_minute,
                total_gaze_aversions
            )
            analysis_results["stats"] = stats
            analysis_results["advice"] = self._generate_advice(stats)

            if plot_output_path:
                plot_result = self._plot_emotions(df_video, output_path=plot_output_path)
                analysis_results["plot_path"] = plot_result
            else:
                self._plot_emotions(df_video)
                analysis_results["plot_path"] = "График был показан (не сохранен)."

            return analysis_results

        except Exception as e:
            # ... (обработка критической ошибки как раньше) ...
            import traceback
            print(f"--- Критическая ошибка во время анализа ---\n{traceback.format_exc()}")
            error_message = f"Критическая ошибка анализа: {str(e)}"
            analysis_results["advice"].append(error_message)
            if not analysis_results.get("stats"): analysis_results["stats"] = self._get_default_stats_on_error(
                error_message)
            return analysis_results
        finally:
            # ... (удаление временных файлов как раньше) ...
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    print(f"Временный аудиофайл {temp_audio_path} удален.")
                except OSError as e_del_audio:
                    print(f"Не удалось удалить {temp_audio_path}: {e_del_audio}")
            if 'processed_ffmpeg_file_to_delete' in locals() and processed_ffmpeg_file_to_delete and os.path.exists(
                    processed_ffmpeg_file_to_delete):
                try:
                    os.remove(processed_ffmpeg_file_to_delete)
                    print(
                        f"Временный видеофайл FFmpeg {processed_ffmpeg_file_to_delete} удален.")
                except OSError as e_del_video:
                    print(f"Не удалось удалить {processed_ffmpeg_file_to_delete}: {e_del_video}")
            end_time = time.time()
            print(f"\nПолный анализ завершен за {end_time - start_time_analysis:.2f} секунд.")

    def _get_default_stats_on_error(self, error_message="Ошибка анализа", video_duration_sec=0):
        return {"stats": error_message}


def main(video_path, plot_output_path=None):
    INTERVIEW_VIDEO = video_path
    DLIB_PREDICTOR = "shape_predictor_68_face_landmarks.dat"

    if not os.path.exists(DLIB_PREDICTOR):
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл предиктора dlib '{DLIB_PREDICTOR}' не найден.")
        print(
            "Пожалуйста, скачайте 'shape_predictor_68_face_landmarks.dat' (например, с dlib.net или github) и поместите в директорию проекта.")
        return
    if not os.path.exists(INTERVIEW_VIDEO):
        print(f"Ошибка: Видеофайл не найден - {INTERVIEW_VIDEO}")
        return

    evaluator = EmotionEvaluator(
        whisper_model_size="medium",
        dlib_predictor_path=DLIB_PREDICTOR,
        device="cpu"
    )

    results = evaluator.analyze_interview(
        INTERVIEW_VIDEO,
        plot_output_path=plot_output_path
    )

    if results:
        print("\n--- Итоговые результаты анализа ---")
        if results.get('stats'):
            print("\nСтатистика:")
            for key, value in results['stats'].items():
                if key == "emotion_distribution" and isinstance(value, dict):
                    print(f"  {key.replace('_', ' ').capitalize()}:")
                    if value:
                        for sub_key, sub_value in value.items(): print(f"    {sub_key.capitalize()}: {sub_value:.1f}%")
                    else:
                        print("    (Нет данных)")
                elif isinstance(value, float):
                    print(f"  {key.replace('_', ' ').capitalize()}: {value:.2f}")
                else:
                    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
        else:
            print("\nСтатистика отсутствует или не была рассчитана.")

        print(f"\nГрафик визуальных эмоций: {results.get('plot_path', 'Не создан/Не сохранен')}")
        print(f"График аудио эмоций: {results.get('audio_plot_path', 'Не создан/Не сохранен')}")
        # print(f"Статистика аудио эмоций: {results.get('audio_stats', {})}") # Можно раскомментировать для детального вывода

        print("\nСоветы:")
        if results.get('advice'):
            for i, tip in enumerate(results['advice']): print(f"{i + 1}. {tip}")
        else:
            print("  (Советы отсутствуют)")

        # Для отладки можно вывести полный словарь результатов
        import json
        print("\n--- Полный результат (JSON для отладки) ---")

        def default_converter(o):
            if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
            raise TypeError

        print(json.dumps(results, indent=2, ensure_ascii=False, default=default_converter))

    else:
        print("Анализ не вернул результатов.")
    return results


# --- Пример использования класса ---
if __name__ == "__main__":
    main(
        "/Users/kristina/IdeaProjects/it-train-diploma/machine-learning-service/soft_skills_analysis/temp/video_analysis_cache/analysis_hvpcb0hw/uploaded_new-1246215995071148178.webm",
        "plots/plot-12462159950711481782.png")
