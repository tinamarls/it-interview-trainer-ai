from moviepy.editor import VideoFileClip
from fastapi import UploadFile
from .logger import logger
import subprocess
import aiofiles

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "temp"))
VIDEO_DIR = os.path.join(BASE_DIR, "video")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")

class FileProcessor:
    def __init__(self, sample_rate: int = 16000, segment_length: float = 3.0, overlap: float = 0.5):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.overlap = overlap

    @staticmethod
    def convert_webm_to_mp4(input_path: str, output_path: str = None) -> str:
        if output_path is None:
            output_path = input_path.rsplit(".", 1)[0] + ".mp4"

        os.system(f"ffmpeg -i {input_path} -c:v libx264 -c:a aac {output_path}")
        return output_path

    @staticmethod
    async def save_upload_file(upload_file: UploadFile) -> str:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        file_path = os.path.join(VIDEO_DIR, upload_file.filename)

        async with aiofiles.open(file_path, "wb") as buffer:
            content = await upload_file.read()
            await buffer.write(content)

        return file_path

    @staticmethod
    def extract_audio_from_video(video_path: str, output_audio_path: str = None) -> str:
        if video_path.endswith(".webm"):
            video_path = FileProcessor.convert_webm_to_mp4(video_path)

        os.makedirs(AUDIO_DIR, exist_ok=True)

        if output_audio_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_audio_path = os.path.join(AUDIO_DIR, f"{base_name}.wav")

        # if output_audio_path is None:
        #     output_audio_path = video_path.rsplit(".", 1)[0] + ".wav"

        logger.info(f"Извлечение аудио из {video_path}")
        logger.debug(f"Целевой путь: {output_audio_path}")

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    output_audio_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")
            logger.info(f"Аудио сохранено в {output_audio_path}")
        except Exception as e:
            logger.error(f"Ошибка при извлечении аудио: {str(e)}")
            raise

        return output_audio_path

    @staticmethod
    def extract_video_only(video_path: str, output_path: str = None) -> str:
        if output_path is None:
            output_path = video_path.rsplit(".", 1)[0] + "_noaudio.mp4"

        video = VideoFileClip(video_path)
        video_without_audio = video.without_audio()
        video_without_audio.write_videofile(output_path, codec="libx264")

        return output_path
