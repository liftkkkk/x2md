# x2md

x2md（X to Markdown）是一个将 PDF/图片/音视频内容整理为 Markdown 或 TXT 的工具集，提供统一命令行，支持 OCR、批处理与音视频辅助能力。

## 功能概述

### 高质量论文提取工具（支持公式、格式保留）
- PDF 转 PNG（高分辨率）
- PNG 批量 OCR 输出 Markdown（DashScope）
- 多页 Markdown 合并
- 批量处理：PDF → PNG → OCR → 合并

### 纯文本提取工具（快速但不保留复杂格式）
- 直接提取 PDF 文本到 TXT（适合简单文本）

### 音视频处理工具
- 视频转 WAV（依赖 ffmpeg）
- 本地语音识别（ModelScope，可选）

## 安装依赖

### Python 依赖包

推荐安装为可执行命令（会提供 `x2md` CLI）：

```bash
pip install -e .
```

按需安装可选能力：

```bash
pip install -e ".[ocr]"   # OCR（DashScope）
pip install -e ".[asr]"   # 语音识别（ModelScope）
pip install -e ".[dev]"   # 开发依赖（ruff/mypy/pytest）
```

如果你只想继续按脚本方式运行，也可以直接安装依赖：

```bash
pip install PyMuPDF dashscope modelscope
```

### 外部依赖（可选）

- **FFmpeg**：用于 `x2md video2audio`，需单独安装并添加到系统 PATH
  - Windows：从 [FFmpeg 官网](https://ffmpeg.org/download.html) 下载并解压，将 bin 目录添加到系统环境变量
  - Linux：`sudo apt-get install ffmpeg`
  - macOS：`brew install ffmpeg`

## 统一命令行（推荐）

```bash
x2md pdf2png "document.pdf" "output_images"
x2md ocr "output_images" "ocr_results"
x2md merge "ocr_results" "merged_document.md"
x2md batch --input-pdf-dir "待处理论文" --output-base-dir "处理结果"
x2md collect --only-merged --source-root "处理结果" --target-dir "."
x2md pdf2txt "document.pdf" -o "output_text.txt"
x2md video2audio "video.mp4" -o "audio.wav"
# ASR（本地模型，二选一：指定模型根目录 或 直接指定模型目录）
x2md asr "audio.wav" -o "recognition_result.json" --model-folder "model_folder"
x2md asr "audio.wav" -o "recognition_result.json" --model-path "model_folder/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
```

## 工作流程

### 高质量论文提取（支持公式、格式保留）

#### 单文件

```bash
x2md pdf2png "paper.pdf" "imgs"
x2md ocr "imgs" "ocr_results"
x2md merge "ocr_results" "paper_merged.md"
```

#### 批量（目录内多 PDF）

```bash
x2md batch --input-pdf-dir "待处理论文" --output-base-dir "处理结果"
x2md collect --only-merged --source-root "处理结果" --target-dir "."
```

### 纯文本快速提取（不保留复杂格式）

```bash
x2md pdf2txt "document.pdf" -o "output_text.txt"
```

### 音视频处理（可选）

```bash
x2md video2audio "video.mp4" -o "audio.wav"
x2md asr "audio.wav" -o "recognition.json" --model-folder "model_folder"
```

## 注意事项

1. **API 密钥设置**：使用 OCR 时，需要在环境变量中设置 `DASHSCOPE_API_KEY`：
   ```bash
   # Windows
   set DASHSCOPE_API_KEY=your_api_key_here
   
   # Linux/macOS
   export DASHSCOPE_API_KEY=your_api_key_here
   ```
2. **图片分辨率**：`x2md pdf2png` 默认 300 DPI，可通过 `--dpi` 调整。
3. **文件命名**：图片默认输出为 `page_001.png` 等，合并与排序依赖页码命名。
4. **OCR 质量**：识别质量受图片清晰度影响，建议使用高分辨率图片。
5. **资源消耗**：处理大型 PDF 可能占用较多内存与磁盘空间。
6. **ASR 本地模型路径**：`x2md asr` 支持直接使用你已下载好的本地模型，不需要每次下载
   - 方式 A（推荐）：指定模型根目录（默认 `./model_folder`）
     ```bash
     x2md asr "audio.wav" --model-folder "C:\path\to\model_folder" -o "recognition.json"
     ```
   - 方式 B：直接指定模型目录（覆盖 `--model-folder`）
     ```bash
     x2md asr "audio.wav" --model-path "C:\path\to\model_folder\iic\speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch" -o "recognition.json"
     ```
   - 可选：用环境变量设置默认路径（后续可省略参数）
     ```bash
     # Windows PowerShell
     $env:X2MD_ASR_MODEL_FOLDER="C:\path\to\model_folder"
     # 或者直接指定模型目录
     $env:X2MD_ASR_MODEL_PATH="C:\path\to\model_folder\iic\speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
     ```
   - 说明：ModelScope 的 pipeline 可能会在首次运行时下载 VAD/标点等依赖模型到用户缓存目录（例如 `~/.cache/modelscope`），首次下载后通常不会重复下载
