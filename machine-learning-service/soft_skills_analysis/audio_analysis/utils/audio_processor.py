from soft_skills_analysis.utils.file_processor import FileProcessor
import whisper
import librosa
from moviepy.editor import VideoFileClip
from fastapi import UploadFile
from typing import  List
from soft_skills_analysis.utils.logger import logger
import shutil
import subprocess
import torchaudio
from pydub import AudioSegment, silence
from pathlib import Path
import torch
import aiofiles

import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "temp"))
SEGMENTS_DIR = os.path.join(BASE_DIR, "segments")

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, segment_length: float = 3.0, overlap: float = 0.5):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.overlap = overlap
        self.file_processor = FileProcessor()

    @staticmethod
    def get_audio_duration(path: str) -> float:
        y, sr = librosa.load(path, sr=None)
        return librosa.get_duration(y=y, sr=sr)

    def transcribe(self, audio_path: str) -> str:
        model = whisper.load_model("base")
        """Транскрибирует аудиофайл с явным указанием языка"""
        try:
            result = model.transcribe(
                audio_path,
                language="ru",  # Явно указываем русский язык
                fp16=False      # Обязательно для CPU
            )
            return result["text"]
        except Exception as e:
            print(f"Ошибка транскрибации: {str(e)}")
            return ""

    def _split_by_silence(self, audio_path: str) -> List[str]:
        logger.info(f"Разделение аудио по паузам: {audio_path}")
        audio = AudioSegment.from_file(audio_path)

        speech_segments = silence.detect_nonsilent(
            audio,
            min_silence_len=500,
            silence_thresh=-40
        )

        logger.debug(f"Найдено сегментов речи: {len(speech_segments)}")

        segments = []
        output_dir = SEGMENTS_DIR
        shutil.rmtree(output_dir, ignore_errors=True)  # Очистка
        os.makedirs(output_dir, exist_ok=True)

        for i, (start, end) in enumerate(speech_segments):
            segment = audio[start:end]
            segment_path = os.path.join(output_dir, f"segment_{i}.wav")
            segment.export(segment_path, format="wav")
            segments.append(segment_path)

        return segments

    def _split_into_chunks(self, audio_path: str, segment_idx: int) -> List[str]:
        audio = AudioSegment.from_file(audio_path).set_frame_rate(self.sample_rate).set_channels(1)
        audio_samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        sr = audio.frame_rate

        segment_samples = int(self.segment_length * sr)
        step_samples = int((self.segment_length - self.overlap) * sr)

        chunks = []
        output_dir = SEGMENTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        chunk_idx = 0

        for i in range(0, len(audio_samples), step_samples):
            chunk = audio_samples[i:i + segment_samples]
            logger.info(chunk)
            duration_sec = len(chunk) / sr
            if len(chunk) < 0.5 * segment_samples:
                logger.info("continue")
                continue

            chunk_path = os.path.join(output_dir, f"segment_{segment_idx}_chunk_{chunk_idx}.wav")
            chunk_idx += 1
            waveform = torch.tensor(chunk).unsqueeze(0)
            torchaudio.save(chunk_path, waveform, sample_rate=sr)
            chunks.append(chunk_path)

        return chunks

    # def process_audio(self, audio_path: str) -> List[str]:
    #     logger.info(f"Обработка аудиофайла: {audio_path}")
    #     all_chunks = []
    #
    #     try:
    #         speech_segments = self._split_by_silence(audio_path)
    #         logger.info(f"Обнаружено {len(speech_segments)} отрывков речи")
    #
    #         for i, seg_path in enumerate(speech_segments):
    #             try:
    #                 audio = AudioSegment.from_file(seg_path)
    #                 duration = len(audio) / 1000.0
    #                 logger.info(f"Сегмент {i + 1}: {seg_path}, длительность: {duration:.2f} сек")
    #             except Exception as e:
    #                 logger.warning(f"Не удалось получить длительность сегмента {seg_path}: {e}")
    #
    #             chunks = self._split_into_chunks(seg_path, i)
    #             logger.info(f"Сегмент {i + 1}: получено {len(chunks)} чанков")
    #             all_chunks.extend(chunks)
    #             Path(seg_path).unlink(missing_ok=True)
    #
    #         logger.info(f"Итоговое число сегментов для анализа: {len(all_chunks)}")
    #     except Exception as e:
    #         logger.error(f"Ошибка при разбиении аудио: {str(e)}")
    #         raise
    #
    #     return all_chunks

    def process_audio(self, audio_path: str) -> List[str]:
        logger.info(f"Обработка аудиофайла: {audio_path}")
        all_chunks = []

        try:
            speech_segments = self._split_by_silence(audio_path)
            logger.info(f"Обнаружено {len(speech_segments)} отрывков речи")

            # for i, seg_path in enumerate(speech_segments):
            #     try:
            #         audio = AudioSegment.from_file(seg_path)
            #         duration = len(audio) / 1000.0
            #         logger.info(f"Сегмент {i + 1}: {seg_path}, длительность: {duration:.2f} сек")
            #     except Exception as e:
            #         logger.warning(f"Не удалось получить длительность сегмента {seg_path}: {e}")
            #
            #     chunks = self._split_into_chunks(seg_path, i)
            #     logger.info(f"Сегмент {i + 1}: получено {len(chunks)} чанков")
            #     all_chunks.extend(chunks)
            #     Path(seg_path).unlink(missing_ok=True)
            #
            # logger.info(f"Итоговое число сегментов для анализа: {len(all_chunks)}")
        except Exception as e:
            logger.error(f"Ошибка при разбиении аудио: {str(e)}")
            raise

        return speech_segments
