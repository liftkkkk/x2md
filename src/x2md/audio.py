from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


MODEL_ID = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
MODEL_REVISION = "v2.0.4"


def _default_local_model_path(model_folder: Union[str, os.PathLike]) -> Path:
    model_folder = Path(model_folder)
    return (
        model_folder
        / "iic"
        / "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    )


def download_asr_model(*, model_folder: Union[str, os.PathLike] = "model_folder") -> Path:
    model_folder = Path(model_folder)
    model_folder.mkdir(parents=True, exist_ok=True)

    from modelscope.hub.snapshot_download import snapshot_download

    downloaded_path = snapshot_download(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(model_folder),
    )
    p = Path(downloaded_path)
    logger.info("模型下载成功: %s", p)
    return p


def transcribe_audio(
    audio_path: Union[str, os.PathLike],
    *,
    output_json: Union[str, os.PathLike] = "recognition_result.json",
    model_folder: Union[str, os.PathLike] = "model_folder",
    model_path: Optional[Union[str, os.PathLike]] = None,
) -> Path:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    output_json = Path(output_json)

    if model_path is None:
        env_model_path = os.environ.get("X2MD_ASR_MODEL_PATH")
        if env_model_path:
            model_path = env_model_path
        else:
            env_model_folder = os.environ.get("X2MD_ASR_MODEL_FOLDER") or os.environ.get("X2MD_MODEL_FOLDER")
            if env_model_folder and str(model_folder) == "model_folder":
                model_folder = env_model_folder
            model_path = _default_local_model_path(model_folder)
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"本地模型目录不存在: {model_path}；请先运行 download-model 或指定 --model-path"
        )

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    inference_pipeline = pipeline(task=Tasks.auto_speech_recognition, model=str(model_path))
    rec_result = inference_pipeline(str(audio_path))
    output_json.write_text(json.dumps(rec_result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("识别结果已保存到: %s", output_json)
    return output_json
