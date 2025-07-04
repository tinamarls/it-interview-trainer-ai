# import cv2
# import numpy as np
# from typing import List, Tuple
# from soft_skills_analysis.utils.logger import logger
# import os
# import pandas as pd
#
# class VideoProcessor:
#     def __init__(self, frame_skip: int = 5, target_size: Tuple[int, int] = (224,224)):
#         self.frame_skip = frame_skip
#         self.target_size = target_size
#
    # def get_open_face_csv(self, video_path: str) -> pd.DataFrame:
    #     """
    #     Анализирует видеофайл с помощью OpenFace и возвращает данные
    #     """
    #     output_dir = "/Users/kristina/IdeaProjects/it-train-diploma/machine-learning-service/soft_skills_analysis/temp/video/open_face_results"
    #     openface_path = "/Users/kristina/IdeaProjects/it-train-diploma/open_face/OpenFace/build/bin/FeatureExtraction"
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     base_name = os.path.splitext(os.path.basename(video_path))[0]
    #     output_csv = os.path.join(output_dir, f"{base_name}.csv")
    #
    #     # Запуск OpenFace с параметрами для определения AU и параметров головы/глаз
    #     cmd = f"{openface_path} -f {video_path} -out_dir {output_dir} -2Dfp -3Dfp -pdmparams -pose -aus -gaze"
    #     os.system(cmd)
    #
    #     if not os.path.exists(output_csv):
    #         raise FileNotFoundError(f"OpenFace не создал файл с результатами: {output_csv}")
    #
    #     try:
    #         df = pd.read_csv(output_csv)
    #     except Exception as e:
    #         raise RuntimeError(f"Не удалось загрузить CSV файл OpenFace {output_csv}: {e}")
    #
    #     if 'confidence' in df.columns:
    #         df_filtered = df[df['confidence'] >= 0.9]
    #         return df_filtered
    #
    #     return pd.read_csv(output_csv)
#
#     def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[float]]:
#         """
#         Извлекает и предобрабатывает кадры из видео
#         Возвращает кадры и временные метки
#         """
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Не удалось открыть видео: {video_path}")
#
#         frames = []
#         timestamps = []
#         frame_idx = 0
#
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             if frame_idx % self.frame_skip == 0:
#                 try:
#                     processed = self._preprocess_frame(frame)
#                     frames.append(processed)
#                     timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
#                 except Exception as e:
#                     logger.error(f"Ошибка обработки кадра {frame_idx}: {e}")
#             frame_idx += 1
#
#         cap.release()
#         return frames, timestamps
#
#     def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
#         """Предобработка кадра для модели"""
#         resized = cv2.resize(frame, self.target_size)
#         rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#         return rgb_frame
#
# # class VideoProcessor:
# #     def __init__(self, frame_skip: int = 5, target_size: Tuple[int, int] = (480, 480)):
# #         self.frame_skip = frame_skip
# #         self.target_size = target_size
# #
# #     def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[float]]:
# #         cap = cv2.VideoCapture(video_path)
# #         if not cap.isOpened():
# #             raise ValueError(f"Не удалось открыть видео: {video_path}")
# #
# #         # Адаптивный frame_skip
# #         fps = cap.get(cv2.CAP_PROP_FPS)
# #         target_fps = 6
# #         self.frame_skip = max(1, int(fps / target_fps))
# #
# #         frames = []
# #         timestamps = []
# #         frame_idx = 0
# #
# #         while cap.isOpened():
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break
# #
# #             if frame_idx % self.frame_skip == 0:
# #                 try:
# #                     processed = self._preprocess_frame(frame)
# #                     frames.append(processed)
# #                     timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
# #                 except Exception as e:
# #                     logger.error(f"Ошибка обработки кадра {frame_idx}: {e}")
# #             frame_idx += 1
# #
# #         cap.release()
# #         logger.info(f"Извлечено {len(frames)} кадров с частотой {target_fps} FPS")
# #         return frames, timestamps
# #
# #     def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
# #         # Проверка яркости и размытости
# #         if np.mean(frame) < 20:
# #             raise ValueError("Кадр слишком темный")
# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
# #             raise ValueError("Кадр слишком размытый")
# #
# #         # Сохранение пропорций
# #         h, w = frame.shape[:2]
# #         scale = min(self.target_size[0] / w, self.target_size[1] / h)
# #         new_w, new_h = int(w * scale), int(h * scale)
# #         resized = cv2.resize(frame, (new_w, new_h))
# #         canvas = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
# #         x_offset = (self.target_size[0] - new_w) // 2
# #         y_offset = (self.target_size[1] - new_h) // 2
# #         canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
# #         rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
# #         return rgb_frame
import cv2
import numpy as np
from typing import List, Tuple, Optional
from soft_skills_analysis.utils.logger import logger
import os
import pandas as pd

class VideoProcessor:
    def __init__(self, frame_skip: int = 10, target_size: Tuple[int, int] = (108, 108)):
        self.frame_skip = frame_skip
        self.target_size = target_size
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def get_open_face_csv(self, video_path: str) -> pd.DataFrame:
        """
        Анализирует видеофайл с помощью OpenFace и возвращает данные
        """
        output_dir = "/Users/kristina/IdeaProjects/it-train-diploma/machine-learning-service/soft_skills_analysis/temp/video/open_face_results"
        # openface_path = "/Users/kristina/IdeaProjects/it-train-diploma/open_face/OpenFace/build/bin/FeatureExtraction"
        openface_path = "/Users/kristina/IdeaProjects/it-train-diploma/OpenFace/OpenFace/build/bin/FeatureExtraction"
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_csv = os.path.join(output_dir, f"{base_name}.csv")

        # Запуск OpenFace с параметрами для определения AU и параметров головы/глаз
        cmd = f"{openface_path} -f {video_path} -out_dir {output_dir} -2Dfp -3Dfp -pdmparams -pose -aus -gaze"
        os.system(cmd)

        if not os.path.exists(output_csv):
            raise FileNotFoundError(f"OpenFace не создал файл с результатами: {output_csv}")

        try:
            df = pd.read_csv(output_csv)
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить CSV файл OpenFace {output_csv}: {e}")

        if 'confidence' in df.columns:
            df_filtered = df[df['confidence'] >= 0.9]
            return df_filtered

        return pd.read_csv(output_csv)

    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[float]]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        timestamps = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_skip == 0:
                try:
                    processed = self._preprocess_frame(frame)
                    if processed is not None:
                        frames.append(processed)
                        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
            frame_idx += 1

        cap.release()
        return frames, timestamps

    def _preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        # Обнаружение лица
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100))

        if len(faces) == 0:
            return None

        # Обрезка и нормализация лица
        (x, y, w, h) = faces[0]
        face = frame[y:y+h, x:x+w]

        # Проверка качества изображения
        if self._is_blurry(face) or self._is_dark(face):
            return None

        # Преобразование для модели
        face = cv2.resize(face, self.target_size)
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    def _is_blurry(self, image: np.ndarray, threshold: int = 100) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

    def _is_dark(self, image: np.ndarray, threshold: int = 50) -> bool:
        return np.mean(image) < threshold