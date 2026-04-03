from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def video_to_wav(
    video_path: Union[str, os.PathLike],
    *,
    output_path: Optional[Union[str, os.PathLike]] = None,
    sample_rate: int = 44100,
    channels: int = 2,
) -> Path:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    output_path = Path(output_path)

    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(output_path),
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("未找到 ffmpeg，请安装并加入 PATH。") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg执行失败: {e.stderr}") from e

    logger.info("音频已保存到: %s", output_path)
    return output_path
