from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, Union

from .utils import RetryPolicy, ensure_dir, sorted_files

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "qwen-vl-ocr-2025-11-20"
DEFAULT_PROMPT = "将双栏论文的内容转换为Markdown格式。"


def ocr_image_to_markdown(
    image_path: Union[str, os.PathLike],
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    retry: RetryPolicy = RetryPolicy(),
) -> str:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY，无法调用 OCR 接口。")

    import dashscope

    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

    messages = [
        {
            "role": "user",
            "content": [{"image": str(image_path)}, {"text": prompt}],
        }
    ]

    delay = retry.initial_delay_s
    last_err: Optional[BaseException] = None
    for attempt in range(1, retry.max_retries + 1):
        try:
            response = dashscope.MultiModalConversation.call(
                api_key=api_key,
                model=model,
                messages=messages,
            )
            return response.output.choices[0].message.content[0]["text"]
        except Exception as e:
            last_err = e
            if attempt >= retry.max_retries:
                break
            logger.warning("API调用失败 (尝试 %s/%s): %s", attempt, retry.max_retries, e)
            logger.info("%ss 后重试...", delay)
            time.sleep(delay)
            delay *= retry.backoff_factor

    raise RuntimeError(f"OCR 调用失败，已重试 {retry.max_retries} 次: {last_err}") from last_err


def ocr_folder_to_markdown(
    input_dir: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    retry: RetryPolicy = RetryPolicy(),
) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是目录: {input_dir}")

    out_dir = ensure_dir(output_dir)
    images = sorted_files(input_dir, suffixes=[".png"])
    logger.info("找到 %s 张图片，开始 OCR...", len(images))

    outputs: list[Path] = []
    for i, img in enumerate(images, 1):
        logger.info("正在处理第 %s/%s 张图片: %s", i, len(images), img.name)
        md = ocr_image_to_markdown(
            img,
            api_key=api_key,
            model=model,
            prompt=prompt,
            retry=retry,
        )
        output_path = out_dir / (img.stem + ".md")
        output_path.write_text(md, encoding="utf-8")
        outputs.append(output_path)
        logger.info("识别完成，结果保存到: %s", output_path.name)

    logger.info("OCR 完成，结果目录: %s", out_dir)
    return outputs
