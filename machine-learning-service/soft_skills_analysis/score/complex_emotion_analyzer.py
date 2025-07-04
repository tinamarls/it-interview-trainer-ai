import os
from typing import Dict, Any, List
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from soft_skills_analysis.utils.logger import logger
from soft_skills_analysis.video_analysis.emotion_detector import VideoEmotionDetector
from soft_skills_analysis.audio_analysis.emotion_detector import AudioEmotionDetector
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

class ComplexEmotionAnalyzer:
    def __init__(self, graph_output_dir: str = "emotion_reports"):
        self.video_detector = VideoEmotionDetector()
        self.audio_detector = AudioEmotionDetector()
        self.graph_output_dir = graph_output_dir
        os.makedirs(self.graph_output_dir, exist_ok=True)

        # Порядок эмоций сверху вниз: enthusiasm, happy, surprise, neutral, sad, disgust, angry, unknown
        self.emotion_to_int = {
            "enthusiasm": 7,
            "happy": 6,
            "surprise": 5,
            "neutral": 4,
            "sad": 3,
            "disgust": 2,
            "angry": 1,
            "unknown": 0
        }
        self.int_to_emotion = {v: k for k, v in self.emotion_to_int.items()}
        self.emotion_normalization_map = {
            "sadness": "sad",
            "happiness": "happy",
            "anger": "angry",
            "enthusiasm": "enthusiasm"
        }
        # Порядок меток для оси Y (от верхней к нижней)
        self.emotion_order = ["enthusiasm", "happy", "surprise", "neutral", "sad", "disgust", "angry", "unknown"]

        try:
            plt.style.use("seaborn-v0_8-whitegrid")  # Try the newer Seaborn style
        except OSError:
            try:
                plt.style.use("seaborn-whitegrid")  # Try the older Seaborn style
            except OSError:
                plt.style.use("whitegrid")          # Fallback to basic Matplotlib
        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300
        })

    def _normalize_emotion_label(self, label: str) -> str:
        if not isinstance(label, str):
            return "unknown"
        label_lower = label.lower()
        return self.emotion_normalization_map.get(label_lower, label_lower)


    def _generate_plotly_graph(
            self,
            video_emotion_sequence: List[str], video_timestamps: List[float],
            audio_emotion_sequence: List[str], audio_timestamps: List[float],
            output_filename: str
    ) -> str:
        """Генерирует интерактивный график с помощью Plotly."""
        has_video = bool(
            video_emotion_sequence and video_timestamps and len(video_emotion_sequence) == len(video_timestamps))
        has_audio = bool(
            audio_emotion_sequence and audio_timestamps and len(audio_emotion_sequence) == len(audio_timestamps))

        if not (has_video or has_audio):
            logger.warning("Объединенный график (Plotly): Нет данных ни для видео, ни для аудио.")
            return ""

        # Настройка цветовой палитры
        video_color = '#1F77B4'  # Глубокий синий
        audio_color = '#FF7F0E'  # Яркий оранжевый
        average_color = '#9467BD'  # Пурпурный

        # Создание подграфиков с большими отступами между ними
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f"<b>Шкала эмоций по видео</b>",
                f"<b>Шкала эмоций по аудио</b>",
                f"<b>Усредненная временная шкала эмоций</b>"
            ),
            vertical_spacing=0.25,  # Увеличиваем отступ между графиками
            specs=[[{"type": "scatter"}],
                   [{"type": "scatter"}],
                   [{"type": "scatter"}]],
        )

        # Настройка оси Y
        y_ticks_values = list(range(len(self.emotion_order)))
        y_ticks_labels = self.emotion_order[::-1]  # Reverse for top-down order

        # 1. График для видео
        if has_video:
            numerical_emotions = [self.emotion_to_int.get(self._normalize_emotion_label(e), self.emotion_to_int["unknown"])
                                  for e in video_emotion_sequence]
            fig.add_trace(
                go.Scatter(
                    x=video_timestamps,
                    y=numerical_emotions,
                    mode='lines+markers',
                    name='Видео',
                    line=dict(color=video_color, width=3),  # Увеличиваем толщину линии
                    marker=dict(size=10, symbol='circle', line=dict(width=2, color='white'))  # Увеличиваем размер маркеров
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(go.Scatter(x=[0], y=[0], mode='text', text=["Нет данных для видео"], textfont=dict(size=16),
                                     showlegend=False), row=1, col=1)

        # 2. График для аудио
        if has_audio:
            numerical_emotions = [self.emotion_to_int.get(self._normalize_emotion_label(e), self.emotion_to_int["unknown"])
                                  for e in audio_emotion_sequence]
            fig.add_trace(
                go.Scatter(
                    x=audio_timestamps,
                    y=numerical_emotions,
                    mode='lines+markers',
                    name='Аудио',
                    line=dict(color=audio_color, width=3),  # Увеличиваем толщину линии
                    marker=dict(size=10, symbol='triangle-up', line=dict(width=2, color='white'))
                    # Увеличиваем размер маркеров
                ),
                row=2, col=1
            )
        else:
            fig.add_trace(go.Scatter(x=[0], y=[0], mode='text', text=["Нет данных для аудио"], textfont=dict(size=16),
                                     showlegend=False), row=2, col=1)

        # 3. График для усредненных эмоций
        if has_video or has_audio:
            all_timestamps = sorted(set(video_timestamps + audio_timestamps))
            if all_timestamps:
                averaged_emotions = []
                for t in all_timestamps:
                    video_emotion = self._get_emotion_at_time(t, video_emotion_sequence, video_timestamps)
                    audio_emotion = self._get_emotion_at_time(t, audio_emotion_sequence, audio_timestamps)
                    video_emotion_norm = self._normalize_emotion_label(video_emotion)
                    audio_emotion_norm = self._normalize_emotion_label(audio_emotion)
                    if video_emotion_norm == "unknown" and audio_emotion_norm == "unknown":
                        averaged_emotions.append("unknown")
                    elif video_emotion_norm == "unknown":
                        averaged_emotions.append(audio_emotion_norm)
                    elif audio_emotion_norm == "unknown":
                        averaged_emotions.append(video_emotion_norm)
                    else:
                        if video_emotion_norm == "neutral" and audio_emotion_norm != "neutral":
                            averaged_emotions.append(audio_emotion_norm)
                        elif audio_emotion_norm == "neutral" and video_emotion_norm != "neutral":
                            averaged_emotions.append(video_emotion_norm)
                        else:
                            averaged_emotions.append(video_emotion_norm)

                numerical_emotions = [self.emotion_to_int.get(e, self.emotion_to_int["unknown"]) for e in averaged_emotions]
                fig.add_trace(
                    go.Scatter(
                        x=all_timestamps,
                        y=numerical_emotions,
                        mode='lines+markers',
                        name='Усредненная',
                        line=dict(color=average_color, width=3),  # Увеличиваем толщину линии
                        marker=dict(size=10, symbol='diamond', line=dict(width=2, color='white'))
                        # Увеличиваем размер маркеров
                    ),
                    row=3, col=1
                )
            else:
                fig.add_trace(go.Scatter(x=[0], y=[0], mode='text', text=["Нет временных меток для усреднения"],
                                         textfont=dict(size=16), showlegend=False), row=3, col=1)
        else:
            fig.add_trace(go.Scatter(x=[0], y=[0], mode='text', text=["Нет данных для усреднения"], textfont=dict(size=16),
                                     showlegend=False), row=3, col=1)

        # Обновление осей Y для всех подграфиков
        for i in range(1, 4):
            fig.update_yaxes(
                tickvals=y_ticks_values,
                ticktext=y_ticks_labels,
                row=i, col=1,
                gridcolor='rgba(220, 220, 220, 0.5)',  # светло-серая сетка
                zeroline=False,
                tickfont=dict(size=14),  # Увеличиваем размер текста для тиков
            )
            fig.update_yaxes(
                title_text="Эмоция",
                row=i, col=1,
                title_font=dict(size=16)  # Увеличиваем размер заголовка оси
            )

            # Обновляем ось X для всех графиков для единообразия
            fig.update_xaxes(
                showgrid=True,
                gridcolor='rgba(220, 220, 220, 0.5)',
                zeroline=False,
                row=i, col=1,
                tickfont=dict(size=14)  # Увеличиваем размер текста для тиков
            )

        # Обновление оси X для нижнего подграфика (добавляем название)
        fig.update_xaxes(
            title_text="Время (секунды)",
            row=3, col=1,
            title_font=dict(size=16)  # Увеличиваем размер заголовка оси
        )

        # Определение темы
        template = "plotly_white"  # Чистая белая тема с минимальным фоновым шумом

        # Общее оформление
        fig.update_layout(
            title={
                'text': f"<b>Анализ эмоций во времени",
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)  # Увеличиваем размер заголовка
            },
            template=template,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=16)  # Увеличиваем размер текста в легенде
            ),
            margin=dict(l=80, r=60, t=120, b=80),  # Увеличиваем отступы от краев
            height=1000,  # Значительно увеличиваем высоту для лучшей читаемости
            width=1200,  # Задаем ширину для лучшего соотношения сторон
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=14)  # Увеличиваем базовый размер шрифта
        )

        # Дополнительная настройка размера и шрифта заголовков подграфиков
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=18, color='#2c3e50')

        # Сохранение в HTML
        graph_path = os.path.join(self.graph_output_dir, output_filename)
        try:
            config = {
                'displayModeBar': True,
                'responsive': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': os.path.basename(output_filename).replace('.html', ''),
                    'height': 1000,
                    'width': 1200,
                    'scale': 2  # High-quality export
                }
            }
            fig.write_html(graph_path, config=config)
            logger.info(f"Интерактивный график сохранен в: {graph_path}")
        except Exception as e:
            logger.error(f"Не удалось сохранить интерактивный график {graph_path}: {e}")
            graph_path = ""

        return graph_path


    def _get_emotion_at_time(self, time: float, emotion_sequence: List[str], timestamps: List[float]) -> str:
        """Возвращает эмоцию для заданного времени, используя ближайший сегмент."""
        if not timestamps or not emotion_sequence or len(timestamps) != len(emotion_sequence):
            return "unknown"

        for i in range(len(timestamps) - 1):
            if timestamps[i] <= time < timestamps[i + 1]:
                return emotion_sequence[i]
        if time >= timestamps[-1]:
            return emotion_sequence[-1]
        return "unknown"

    def _create_overall_summary(self, video_analysis: Dict, audio_analysis: Dict) -> Dict:
        """Создает упрощенный отчет с требуемыми параметрами."""
        summary = {}

        # 1. Преобладающая эмоция (видео)
        video_dom_emotion = self._normalize_emotion_label(video_analysis.get('dominant_emotion', 'unknown'))
        summary['dominant_emotion_video'] = video_dom_emotion

        # 2. Преобладающая эмоция (аудио)
        audio_avg_emotions = audio_analysis.get('average_emotions', {})
        audio_dom_emotion = self._normalize_emotion_label(
            max(audio_avg_emotions, key=audio_avg_emotions.get, default='unknown') if audio_avg_emotions else 'unknown'
        )
        summary['dominant_emotion_audio'] = audio_dom_emotion

        # 3. Частота смены эмоций (видео)
        video_changes = video_analysis.get('emotion_change_frequency', 0.0)
        summary['video_emotion_changes_per_minute'] = round(video_changes, 2)

        # 4. Частота смены эмоций (аудио)
        audio_changes = audio_analysis.get('emotion_changes_per_minute', 0.0)
        summary['audio_emotion_changes_per_minute'] = round(audio_changes, 2)

        # 5. Средние параметры
        # Усредненное распределение эмоций
        video_dist = video_analysis.get('emotion_distribution', {})
        audio_dist = audio_analysis.get('average_emotions', {})
        all_emotions = set(video_dist.keys()) | set(audio_dist.keys())
        average_dist = {}
        for emotion in all_emotions:
            norm_emotion = self._normalize_emotion_label(emotion)
            video_val = video_dist.get(emotion, 0.0)
            audio_val = audio_dist.get(emotion, 0.0)
            average_dist[norm_emotion] = (video_val + audio_val) / 2
        summary['average_emotion_distribution'] = {k: round(v, 4) for k, v in average_dist.items()}

        # Общая стабильность
        video_stability = 1 - min(video_changes / 15, 1.0)  # Нормализация: 15 изменений/мин = 0 стабильности
        audio_stability = audio_analysis.get('emotion_dynamics', {}).get('emotion_stability', 0.0)
        summary['overall_stability'] = round((video_stability + audio_stability) / 2, 2)

        return summary

    def analyze_video_response(self, video_path: str, extracted_audio_path: str, output_format: str = "html") -> Dict[str, Any]:
        if not os.path.exists(video_path):
            logger.error(f"Видеофайл не найден: {video_path}")
            return {"error": f"Видеофайл не найден: {video_path}"}
        if not os.path.exists(extracted_audio_path):
            logger.error(f"Аудиофайл не найден: {extracted_audio_path}")
            return {"error": f"Аудиофайл не найден: {extracted_audio_path}"}

        logger.info(f"Начало комплексного анализа для видео: {video_path} и аудио: {extracted_audio_path}")

        video_analysis_results = self.video_detector.analyze_video(video_path)
        audio_analysis_results = self.audio_detector.analyze_emotion(extracted_audio_path)

        base_name = os.path.splitext(os.path.basename(video_path))[0]

        # Генерация объединенного интерактивного графика
        graph_path = self._generate_plotly_graph(
            video_analysis_results.get('emotion_sequence', []),
            video_analysis_results.get('timestamps', []),
            audio_analysis_results.get('emotion_sequence', []),
            audio_analysis_results.get('segment_timestamps', []),
            f"emotion_timelines_{base_name}.{output_format.lower()}"
        )

        comprehensive_summary = self._create_overall_summary(video_analysis_results, audio_analysis_results)

        logger.info(f"Комплексный анализ для {video_path} завершен.")
        return {
            "comprehensive_assessment": comprehensive_summary,
            "emotion_timeline_graph_file": graph_path or None
        }

    def main(self, actual_test_video_path, actual_test_audio_path):
        analyzer = ComplexEmotionAnalyzer(graph_output_dir="final_emotion_reports_plotly")
        logger.info(f"\n--- Запуск комплексного анализа для видео '{actual_test_video_path}' и аудио '{actual_test_audio_path}' ---")

        analysis_report = analyzer.analyze_video_response(
            actual_test_video_path,
            actual_test_audio_path,
            output_format="html"
        )

        logger.info("\n--- Отчет о комплексном анализе ---")
        import json
        def pretty_print_report(report):
            def default_serializer(obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, (Counter, defaultdict)): return dict(obj)
                if hasattr(obj, '__dict__'): return str(obj)
                try: return str(obj)
                except: return repr(obj)
            print(json.dumps(report, indent=2, ensure_ascii=False, default=default_serializer))

        pretty_print_report(analysis_report)

        if analysis_report.get("emotion_timeline_graph_file"):
            logger.info(f"Интерактивный график сохранен в: {analysis_report['emotion_timeline_graph_file']}")
        return analysis_report

if __name__ == '__main__':
    video_path = "../temp/video_analysis_cache/analysis_9ssctehf/uploaded-6268408549339226737.webm"
    audio_path = "../temp/video_analysis_cache/analysis_9ssctehf/uploaded-6268408549339226737.wav"
    complex_emotion_analyzer = ComplexEmotionAnalyzer()
    complex_emotion_analyzer.main(video_path, audio_path)
