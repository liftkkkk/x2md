# 🚀 x2md: Turn Unstructured Documents into Clean Markdown in One Click

**English** | [简体中文](./README.zh-CN.md)

`x2md` (X to Markdown) is a one-stop intelligent document-processing toolkit. It converts complex **PDFs, academic papers, images, and audio/video** into well-structured, easy-to-edit **Markdown** or **TXT**.

Whether you are preprocessing an **LLM knowledge base**, organizing **academic papers**, or automating **meeting notes**, `x2md` delivers enterprise-grade productivity.

---

## 🌟 Core Value

### 1. Lab-grade paper reconstruction
More than text extraction. For academic papers, the "high-DPI rendering + intelligent OCR" pipeline preserves as much as possible:
* **Math formulas** (LaTeX-friendly)
* **Complex layout and hierarchy**
* **Seamless multi-page merging**

### 2. Industrial-grade batch processing
Stop handling files one by one:
* **Resumable runs**: interrupted? No problem — on restart, completed pages are skipped automatically.
* **Auditable manifest**: the generated `manifest.jsonl` computes success rate, elapsed time, and API cost.
* **Error tracing**: failed samples are written to disk, so recognition hotspots are easy to spot.

### 3. Full multimodal coverage
Beyond documents, it is also your media assistant:
* **Video/audio to text**: built-in ASR quickly drafts meeting minutes.
* **Extremely lightweight**: a plain-text fast mode processes hundred-page documents in seconds.

---

## 🛠️ Quick Start

### 1. Install
Pick the install size you need:
```bash
pip install x2md              # Basic: PDF plain-text conversion
pip install "x2md[ocr]"       # Enhanced: formula reconstruction, image OCR (recommended)
pip install "x2md[asr]"       # Full: adds audio/video transcription
```

### 2. Convert a complex paper in three steps
Three simple commands turn a PDF paper into perfect Markdown:
```bash
# 1. Render to high-DPI images
x2md pdf2png "paper.pdf" "imgs"

# 2. Intelligent OCR (requires a DashScope API key)
x2md ocr "imgs" "results" --keep-going

# 3. Merge automatically
x2md merge "results" "paper_final.md"
```

---

## 💡 Common Scenarios

### 📂 Batch-process an entire folder
For building knowledge bases or processing large historical archives:
```bash
# Batch-process the "todo" directory; results are archived under "output"
x2md batch --input-pdf-dir "./my_papers" --output-base-dir "./output"

# Collect all successfully merged Markdown files into one folder
x2md collect --only-merged --source-root "./output" --target-dir "./final_md"
```

### 🎙️ Video/audio organization
Extract audio from video and transcribe it to text quickly:
```bash
x2md video2audio "meeting.mp4" -o "voice.wav"
x2md asr "voice.wav" -o "summary.json" --model-folder "./models"
```

### 📊 Quality evaluation & serving
* **Evaluation**: `x2md eval` compares recognition results against ground truth and computes accuracy.
* **Cost analysis**: `x2md report` generates a time-and-cost report in one command.
* **API service**: `x2md serve` instantly turns the toolkit's capabilities into backend API endpoints.

---

## ⚠️ Tips

1.  **API config**: before using intelligent OCR, make sure the environment variable is set:
    * `export DASHSCOPE_API_KEY=your_key`
2.  **Multimedia dependencies**: audio/video processing requires `FFmpeg` installed on the system.
3.  **Local ASR**: for speed, point `--model-folder` to a locally downloaded model.

---

## 📈 Why choose x2md?

| Feature | Traditional OCR tools | x2md |
| :--- | :--- | :--- |
| **Formula support** | Poor, often garbled | **Excellent, LaTeX-friendly** |
| **Throughput** | Single-threaded, fragile | **High-concurrency batch, resumable** |
| **Transparency** | Black box | **Detailed cost and success-rate manifests** |
| **Modality** | Documents only | **PDF / images / audio & video, fully covered** |

---

**Start using x2md now and unleash your document productivity!**
