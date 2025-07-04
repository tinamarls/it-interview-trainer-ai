import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import torch
import torchaudio.transforms as T
from pydub import AudioSegment
from scipy.signal import savgol_filter
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification

from soft_skills_analysis.audio_analysis.speech_features import get_f0_variability
from soft_skills_analysis.audio_analysis.speech_features import get_filler_rate


class AudioEmotionAnalyzer:
    """
    Класс для анализа эмоций в аудиозаписи интервью (с поддержкой русского языка).

    Использует предобученную модель для распознавания эмоций в речи,
    разбивает аудио на сегменты, анализирует каждый сегмент и предоставляет
    график изменения эмоций по времени и статистику.
    """

    def __init__(self, original_audio_path: str, segments: list,
                 # model_name: str = "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned",
                 model_name: str = "Aniemore/wavlm-emotion-russian-resd",
                 plot_output_path: str = None, transcript: str = None):
        """
        Инициализация анализатора.

        Args:
            original_audio_path (str): Путь к оригинальному полному аудиофайлу.
                                       Необходим для извлечения аудиоданных
                                       для каждого сегмента.
            segments (list): Список объектов сегментов, каждый из которых
                             должен иметь атрибуты 'start' и 'end' в секундах.
                             Например, сегменты, полученные из faster_whisper.
            model_name (str): Имя предобученной модели Hugging Face для SER.
                              По умолчанию используется модель, обученная на русской речи.
            plot_output_path (str, optional): Путь для сохранения графика.
                                              Если None, график будет отображен.
        """
        self.plot_output_path = plot_output_path
        self.transcript = transcript

        required_libraries = [pd, plt, pipeline, torch, sf]
        if AudioSegment is not None:
            required_libraries.append(AudioSegment)

        if not all(lib is not None for lib in required_libraries):
            missing = [name for name, lib in zip(['pandas', 'matplotlib.pyplot', 'pydub', 'transformers.pipeline', 'torch', 'soundfile'], required_libraries) if lib is None]
            raise ImportError(f"Не все необходимые библиотеки установлены или доступны: {', '.join(missing)}. Пожалуйста, установите их.")

        if not os.path.exists(original_audio_path):
            raise FileNotFoundError(f"Оригинальный аудиофайл не найден: {original_audio_path}")

        if not segments:
            raise ValueError("Список сегментов не может быть пустым.")

        if AudioSegment is None:
            raise ImportError("Библиотека 'pydub' не установлена. Она необходима для загрузки и нарезки оригинального аудио.")


        self.original_audio_path = original_audio_path
        self.segments = segments # Список сегментов (например, от faster_whisper)
        self.model_name = model_name
        self.results_df = None
        self.sampling_rate = None  # Будет определена при загрузке аудио
        self.original_audio = None # Объект pydub.AudioSegment для полного аудио

        # Загрузка модели и процессора один раз
        try:
            print(f"Загрузка модели '{model_name}'...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(model_name)
            # Переносим модель на GPU, если доступно
            self.device = 0 if torch.cuda.is_available() else -1
            self.classifier = pipeline("audio-classification", model=self.model,
                                       feature_extractor=self.feature_extractor, device=self.device)
            self.emotion_labels = list(self.model.config.id2label.values())  # Получаем метки эмоций из конфига модели
            print(f"Модель загружена. Обнаруженные эмоции: {self.emotion_labels}")

        except Exception as e:
            print(f"Ошибка при загрузке модели {model_name}: {e}")
            print("Проверьте имя модели или ваше интернет-соединение.")
            self.classifier = None  # Устанавливаем None, чтобы предотвратить дальнейшее использование

    def _load_original_audio(self) -> AudioSegment:
        """
        Внутренний метод для загрузки полного оригинального аудиофайла.
        """
        if AudioSegment is None:
            raise ImportError("Библиотека 'pydub' не установлена.")
        if self.original_audio is not None:
            return self.original_audio # Уже загружено

        try:
            audio = AudioSegment.from_file(self.original_audio_path)
            self.sampling_rate = audio.frame_rate
            self.original_audio = audio
            print(
                f"Оригинальное аудио загружено. Длительность: {len(audio) / 1000:.2f} сек, Частота дискретизации: {self.sampling_rate} Гц")
            return self.original_audio
        except Exception as e:
            raise IOError(
                f"Ошибка при загрузке оригинального аудиофайла: {e}. Убедитесь, что формат поддерживается pydub и установлен ffmpeg.")


    def analyze_segment(self, segment_audio: AudioSegment, segment_start_ms: int, segment_end_ms: int) -> dict:
        """
        Анализирует эмоции в одном сегменте аудио (pydub.AudioSegment).

        Args:
            segment_audio (AudioSegment): Сегмент аудио pydub.
            segment_start_ms (int): Время начала сегмента в миллисекундах (из whisper segment).
            segment_end_ms (int): Время окончания сегмента в миллисекундах (из whisper segment).

        Returns:
            dict: Словарь с вероятностями эмоций и временными метками.
        """

        try:
            if self.classifier is None:
                raise RuntimeError("Модель классификации эмоций не загружена.")

            # Проверяем, что сегмент не пустой
            if len(segment_audio) == 0:
                print(f"Предупреждение: Пустой сегмент на {segment_start_ms}-{segment_end_ms} мс. Пропускаем анализ.")
                return {
                    'start_time_ms': segment_start_ms,
                    'end_time_ms': segment_end_ms,
                    'duration_ms': segment_end_ms - segment_start_ms,
                    'error': 'Empty segment'
                }

            # Конвертируем pydub -> numpy -> torch.tensor
            samples = np.array(segment_audio.get_array_of_samples()).astype(np.float32)

            # Учитываем количество каналов (pydub возвращает интерливинг)
            if segment_audio.channels > 1:
                samples = samples.reshape((-1, segment_audio.channels))
                samples = samples.mean(axis=1)  # Преобразуем в моно

            # Нормализуем значения (если int16 → [-1.0, 1.0])
            if segment_audio.sample_width == 2:
                samples /= 2 ** 15
            elif segment_audio.sample_width == 4:
                samples /= 2 ** 31

            waveform = torch.tensor(samples).unsqueeze(0)  # (1, time)

            original_sr = segment_audio.frame_rate
            target_sr = self.feature_extractor.sampling_rate

            if original_sr != target_sr:
                resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
                waveform = resampler(waveform)

            # Применяем feature extractor
            inputs = self.feature_extractor(
                [waveform.squeeze(0).numpy()],  # передаём список одномерных массивов
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True
            )

            # Отправляем на нужное устройство
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits

            scores = torch.nn.functional.softmax(logits, dim=-1)[0]
            emotion_scores = {self.model.config.id2label[i]: float(scores[i]) for i in range(len(scores))}

            # Добавляем временные метки из предоставленного сегмента
            emotion_scores['start_time_ms'] = segment_start_ms
            emotion_scores['end_time_ms'] = segment_end_ms
            emotion_scores['duration_ms'] = segment_end_ms - segment_start_ms

            return emotion_scores
        except Exception as e:
            print(f"Ошибка при анализе сегмента {segment_start_ms}-{segment_end_ms} мс: {e}")
            return {
                'start_time_ms': segment_start_ms,
                'end_time_ms': segment_end_ms,
                'duration_ms': segment_end_ms - segment_start_ms,
                'error': str(e)
            }

    def analyze_audio(self) -> pd.DataFrame:
        """
        Выполняет анализ предоставленных аудиосегментов.

        Returns:
            pd.DataFrame: DataFrame с результатами анализа каждого сегмента.
                          Колонки: 'start_time_ms', 'end_time_ms', 'duration_ms'
                          и колонки для каждой эмоции с их вероятностями.
                          Сохраняет результат в self.results_df.
        """

        if self.classifier is None:
            print("Анализ не может быть выполнен: модель классификации эмоций не загружена.")
            self.results_df = pd.DataFrame()
            return self.results_df

        print("Начинаю анализ предоставленных сегментов...")

        # Загружаем оригинальное аудио один раз
        try:
            original_audio = self._load_original_audio()
        except IOError as e:
            print(f"Не удалось загрузить оригинальное аудио: {e}")
            self.results_df = pd.DataFrame()
            return self.results_df


        results = []
        # Убедимся, что список всех возможных эмоций получен корректно
        all_possible_emotion_labels = self.emotion_labels
        num_segments = len(self.segments)

        micro_segment_duration_ms = 1000 # Длительность микро-сегмента, например, 1 секунда
        min_chunk_duration_ms = 200      # Минимальная длительность последнего микро-сегмента для анализа

        for i, seg_info in enumerate(self.segments):
            whisper_start_ms = int(seg_info['start'] * 1000)
            whisper_end_ms = int(seg_info['end'] * 1000)
            whisper_duration_ms = whisper_end_ms - whisper_start_ms

            # Пропускаем очень короткие сегменты Whisper
            if whisper_duration_ms < min_chunk_duration_ms:
                print(f"Пропускаем короткий сегмент {i+1}/{num_segments} ({whisper_start_ms}-{whisper_end_ms} мс). Длительность: {whisper_duration_ms} мс.")
                continue

            print(f"Обработка сегмента Whisper {i+1}/{num_segments} ({whisper_start_ms} - {whisper_end_ms} мс)")

            # --- Внутренний цикл для микро-сегментов ---
            current_chunk_start_ms = whisper_start_ms
            chunk_index = 0

            while current_chunk_start_ms < whisper_end_ms:
                # Конец текущего микро-сегмента - либо старт + длительность, либо конец сегмента Whisper (если он короче)
                current_chunk_end_ms = min(current_chunk_start_ms + micro_segment_duration_ms, whisper_end_ms)

                # Проверяем, что текущий кусок (особенно последний) достаточной длительности
                chunk_actual_duration_ms = current_chunk_end_ms - current_chunk_start_ms
                if chunk_actual_duration_ms < min_chunk_duration_ms:
                    # Если оставшаяся часть слишком мала, чтобы быть проанализированной как отдельный кусок
                    break # Завершаем внутренний цикл для этого сегмента Whisper

                chunk_audio = original_audio[current_chunk_start_ms:current_chunk_end_ms]

                # Анализируем микро-сегмент
                chunk_results = self.analyze_segment(chunk_audio, current_chunk_start_ms, current_chunk_end_ms)

                if chunk_results:
                    # Опционально: добавляем информацию о родительском сегменте Whisper
                    chunk_results['whisper_segment_index'] = i
                    chunk_results['whisper_segment_start_ms'] = whisper_start_ms
                    chunk_results['whisper_segment_end_ms'] = whisper_end_ms

                    results.append(chunk_results)
                    if 'error' not in chunk_results:
                        print(f"  Проанализирован микро-сегмент {chunk_index+1} ({current_chunk_start_ms}-{current_chunk_end_ms} мс)")
                    else:
                        print(f"  Ошибка анализа микро-сегмента {chunk_index+1} ({current_chunk_start_ms}-{current_chunk_end_ms} мс): {chunk_results['error']}")

                chunk_index += 1
                # *** Переходим к следующему старту БЕЗ учета перекрытия ***
                current_chunk_start_ms += micro_segment_duration_ms

        # for i, seg_info in enumerate(self.segments):
        #     start_ms = int(seg_info['start'] * 1000)
        #     end_ms = int(seg_info['end'] * 1000)
        #
        #     # Извлекаем аудио для текущего сегмента из полного аудио
        #     segment_audio = original_audio[start_ms:end_ms]
        #
        #     segment_results = self.analyze_segment(segment_audio, start_ms, end_ms)
        #
        #     if segment_results: # Добавляем результат, даже если есть ошибка, чтобы отследить
        #         results.append(segment_results)
        #         if 'error' not in segment_results:
        #             print(f"Проанализирован сегмент {i+1}/{num_segments} ({start_ms} - {end_ms} мс)")
        #         else:
        #             print(f"Ошибка анализа сегмента {i+1}/{num_segments} ({start_ms} - {end_ms} мс): {segment_results['error']}")

        if not results:
            print("Анализ не дал результатов.")
            self.results_df = pd.DataFrame()
            return self.results_df

        self.results_df = pd.DataFrame(results)

        # Добавляем колонки для эмоций, которые могли не встретиться ни в одном успешном сегменте
        missing_in_df_cols = [col for col in all_possible_emotion_labels if col not in self.results_df.columns]

        for label in missing_in_df_cols:
            self.results_df[label] = 0.0  # Добавляем отсутствующие колонки эмоций

        # Заполняем NaN в колонках эмоций нулями (может возникнуть из-за сегментов с ошибками анализа)
        # Делаем это для *всех* строк DataFrame
        for col in all_possible_emotion_labels:
            if col in self.results_df.columns:
                self.results_df[col] = self.results_df[col].fillna(0.0)


        # Порядок колонок: сначала время, потом эмоции, потом ошибка
        time_cols = ['start_time_ms', 'end_time_ms', 'duration_ms']
        # Список колонок эмоций из всех возможных
        emotion_cols_final = [col for col in all_possible_emotion_labels if col in self.results_df.columns]
        # Сортируем колонки эмоций для консистентности
        emotion_cols_final.sort()
        # Добавляем колонку 'error', если она существует
        other_cols = [col for col in self.results_df.columns if col not in time_cols + emotion_cols_final]
        # Убедимся, что 'error' всегда последняя, если есть
        if 'error' in other_cols:
            other_cols.remove('error')
            other_cols.append('error')

        column_order = time_cols + emotion_cols_final + other_cols
        self.results_df = self.results_df[column_order]

        print("Анализ завершен.")
        return self.results_df

    def get_results(self) -> pd.DataFrame:
        """
        Возвращает DataFrame с результатами анализа.

        Returns:
            pd.DataFrame: DataFrame с результатами анализа каждого сегмента.
                          None, если анализ еще не проводился.
        """

        if self.results_df is None:
            print("Анализ еще не проводился. Вызовите analyze_audio() сначала.")
        return self.results_df

    def plot_smoothed_dominant_emotion_curve(self, figsize=(15, 5), smooth_window: int = 5, polyorder: int = 2):
        """
        Строит сглаженную кривую доминирующей эмоции по времени, используя числовое кодирование эмоций.
        Использует только сегменты, которые были проанализированы без ошибок.

        Args:
            figsize (tuple): Размер фигуры.
            smooth_window (int): Размер окна сглаживания (должно быть нечетным и >= polyorder + 2).
                                 Рекомендуется 5 или больше.
            polyorder (int): Порядок полинома для фильтрации. Рекомендуется 2 или 3.
        """
        if self.results_df is None or self.results_df.empty:
            print("Нет данных для построения графика.")
            return

        # Фильтруем только успешные сегменты для построения графика
        if 'error' in self.results_df.columns:
            df = self.results_df[
                self.results_df['error'].isna() | (self.results_df['error'] == 'Empty segment')
                ].copy()
        else:
            df = self.results_df.copy()

        if df.empty:
            print("Нет успешных сегментов для построения графика.")
            return

        df['center_time_sec'] = (df['start_time_ms'] + df['end_time_ms']) / 2000.0

        emotion_cols = [emo for emo in self.emotion_labels if emo in df.columns]
        if not emotion_cols:
            print("Нет данных по эмоциям в успешных сегментах для построения графика.")
            return

        # Перевод эмоций на русский (пример, может отличаться в зависимости от модели)
        emotion_rus = {
            'neutral': 'Нейтральная',
            'anger': 'Злость',
            'happiness': 'Счастье',
            'sadness': 'Грусть',
            'fear': 'Страх',
            'enthusiasm': 'Энтузиазм',
            'disgust': 'Отвращение'
            # Добавьте другие эмоции, если ваша модель их распознает
        }
        # Используем метки из self.emotion_labels в качестве основы, чтобы не потерять те, которых нет в emotion_rus
        # Если метки модели не в emotion_rus, используем их как есть
        emotion_rus_mapping = {label: emotion_rus.get(label, label) for label in self.emotion_labels}


        # Определение доминирующей эмоции только среди emotion_cols
        df['dominant_emotion'] = df[emotion_cols].idxmax(axis=1)

        # Определяем порядок эмоций для оси Y на основе всех возможных эмоций модели
        # Сначала те, что есть в emotion_rus_mapping, потом остальные
        # ordered_emotions = sorted(emotion_rus_mapping.keys(), key=lambda x: emotion_rus_mapping[x])

        ordered_emotions = ['anger', 'disgust', 'fear', 'sadness', 'neutral', 'happiness', 'enthusiasm']

        # Фильтруем только те эмоции, которые есть в наших данных
        ordered_emotions = [emo for emo in ordered_emotions if emo in emotion_cols]

        # Если есть эмоции в данных, которых нет в нашем списке, добавляем их в конец
        missing_emotions = [emo for emo in emotion_cols if emo not in ordered_emotions]
        ordered_emotions.extend(missing_emotions)

        # Эмоции в числовой код по определенному порядку
        emotion_to_int = {emo: i for i, emo in enumerate(ordered_emotions)}
        # ИСПРАВЛЕНИЕ: переменные emo и i должны быть поменяны местами при распаковке
        int_to_emotion_rus = {i: emotion_rus_mapping[emo] for emo, i in emotion_to_int.items()}

        df['dominant_emotion_id'] = df['dominant_emotion'].map(emotion_to_int)

        x = df['center_time_sec'].values
        y = df['dominant_emotion_id'].values

        # Проверка на достаточное количество точек для сглаживания
        if len(y) < smooth_window or smooth_window % 2 == 0 or smooth_window < polyorder + 2:
            print(f"Недостаточно точек ({len(y)}) или некорректные параметры сглаживания (window={smooth_window}, polyorder={polyorder}). Сглаживание не применено.")
            y_smooth = y
        else:
            try:
                # Убедитесь, что savgol_filter импортирован, например: from scipy.signal import savgol_filter
                from scipy.signal import savgol_filter 
                y_smooth = savgol_filter(y, smooth_window, polyorder)
            except Exception as e:
                print(f"Ошибка при сглаживании: {e}. Используются несглаженные данные.")
                y_smooth = y


        # Построение графика
        # Убедитесь, что matplotlib.pyplot и seaborn импортированы, например:
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=figsize)
        sns.set(style="whitegrid")

        # Рисуем линию
        plt.plot(x, y_smooth, label='Кривая вероятности эмоции', linewidth=2, color='royalblue')
        # Опционально: отобразить исходные точки
        # plt.scatter(x, y, c='darkorange', label='Исходные эмоции', zorder=5)


        # Оформление оси Y
        yticks = list(int_to_emotion_rus.keys())
        ylabels = [int_to_emotion_rus[i] for i in yticks]
        # Убедимся, что yticks соответствуют диапазону значений y_smooth
        min_y = min(yticks) if yticks else 0
        max_y = max(yticks) if yticks else 1
        plt.ylim(min_y - 0.5, max_y + 0.5) # Добавляем небольшой отступ по краям

        plt.yticks(yticks, ylabels)

        plt.xlabel("Время (секунды)", fontsize=12)
        plt.ylabel("Доминирующая эмоция", fontsize=12)
        plt.title("Кривая доминирующей эмоции", fontsize=14)
        # plt.legend() # Убираем легенду, т.к. есть только одна линия
        plt.grid(True)
        plt.tight_layout()

        if self.plot_output_path:
            try:
                plt.savefig(self.plot_output_path)
                print(f"График сохранен: {self.plot_output_path}")
                plt.close()
                return self.plot_output_path
            except Exception as e:
                print(f"Ошибка при сохранении графика по пути {self.plot_output_path}: {e}")
                plt.show() # Показываем, если не удалось сохранить
                return None

        else:
            plt.show()
            return None

    def get_statistics(self) -> dict:
        """
        Рассчитывает статистику по эмоциям, исключая сегменты с ошибками анализа.

        Returns:
            dict: Словарь со статистикой:
                  - 'average_probabilities': Средние вероятности каждой эмоции (только для успешных сегментов).
                  - 'dominant_emotion_overall': Наиболее часто встречающаяся доминирующая эмоция (только для успешных сегментов).
                  - 'dominant_emotion_duration_sec': Общая длительность (в секундах) для каждого типа эмоции,
                                                     в течение которой она была доминирующей (только для успешных сегментов).
                  - 'segment_errors': Количество сегментов, которые не удалось проанализировать на эмоции.
                  - 'total_segments': Общее количество предоставленных сегментов.
        """
        if pd is None:
            raise ImportError("Библиотека 'pandas' не установлена.")

        if self.results_df is None or self.results_df.empty:
            print("Нет данных для расчета статистики. Вызовите analyze_audio() сначала.")
            return {'total_segments': len(self.segments), 'segment_errors': len(self.segments), 'average_probabilities': {}, 'dominant_emotion_overall': "N/A", 'dominant_emotion_duration_sec': {}}

        # Успешные сегменты для расчета статистики по эмоциям
        df_success = self.results_df.copy()


        stats = {
            'total_segments': len(self.segments),
            'average_probabilities': {},
            'dominant_emotion_overall': "N/A (No successful segments)",
            'dominant_emotion_duration_sec': {}
        }

        if df_success.empty:
            print("Нет успешных результатов анализа для расчета статистики по эмоциям.")
            return stats

        # Берем колонки эмоций только из успешных сегментов, которые действительно присутствуют
        emotion_cols_present_in_success = [col for col in self.emotion_labels if col in df_success.columns]

        if not emotion_cols_present_in_success:
            print("Нет данных по эмоциям в успешных сегментах для статистики.")
            return stats

        # Средние вероятности (только для успешных сегментов)
        stats['average_probabilities'] = df_success[emotion_cols_present_in_success].mean().to_dict()

        # Доминирующая эмоция для каждого успешного сегмента
        df_success['dominant_emotion'] = df_success[emotion_cols_present_in_success].idxmax(axis=1)

        # Наиболее часто встречающаяся доминирующая эмоция (только среди успешных сегментов)
        dominant_overall_mode = df_success['dominant_emotion'].mode().tolist()
        stats['dominant_emotion_overall'] = dominant_overall_mode[0] if dominant_overall_mode else "N/A"

        # Общая длительность для каждой доминирующей эмоции (только для успешных сегментов)
        df_success['duration_sec'] = df_success['duration_ms'] / 1000.0
        dominant_duration = df_success.groupby('dominant_emotion')['duration_sec'].sum().to_dict()
        stats['dominant_emotion_duration_sec'] = dominant_duration
        stats["f0_variability"] = get_f0_variability(self.original_audio_path)
        stats["filler_rate"] = get_filler_rate(self.transcript, len(self.original_audio))

        return stats