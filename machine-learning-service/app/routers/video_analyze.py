import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from soft_skills_analysis.score.emotion_eval import EmotionEvaluator
from soft_skills_analysis.utils.file_processor import FileProcessor

router = APIRouter()

# Настройка логгера для отслеживания процесса
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_processor = FileProcessor()
emotion_evaluator = EmotionEvaluator()


@router.post("/analyze_my_new")
async def analyze_my_new(video: UploadFile = File(...)):
    base_temp_dir = Path("soft_skills_analysis/temp/video_analysis_cache")
    base_temp_dir.mkdir(parents=True, exist_ok=True)

    # Создаем уникальную временную поддиректорию для этого запроса
    temp_dir_path = Path(tempfile.mkdtemp(prefix="analysis_", dir=base_temp_dir))

    video_path = temp_dir_path / video.filename

    # 1. Сохраняем загруженное видео во временный файл
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    logger.info(f"Видео сохранено: {video_path}")

    return emotion_evaluator.analyze_interview(str(video_path),
                                               str(temp_dir_path) + "/" + video.filename.split(".")[0] + "_plot.png")

