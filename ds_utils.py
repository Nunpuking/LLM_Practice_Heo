# Sources are borrowed from https://github.com/microsoft/DeepSpeedExamples/blob/master/training/HelloDeepSpeed/train_bert_ds.py#L285
import os
import logging
import loguru
from typing import List

logger = loguru.logger

def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def log_dist(message: str,
             ranks: List[int] = [],
             level: int = logging.INFO) -> None:
    """Log messages for specified ranks only"""
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank in ranks:
        if level == logging.INFO:
            logger.info(f'[Rank {my_rank}] {message}')
        if level == logging.ERROR:
            logger.error(f'[Rank {my_rank}] {message}')
        if level == logging.DEBUG:
            logger.debug(f'[Rank {my_rank}] {message}')
