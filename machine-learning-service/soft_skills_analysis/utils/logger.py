# soft_skills_analysis/utils/logger.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("soft_skills_logger")
