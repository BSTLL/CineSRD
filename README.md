# CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
<!-- [![Arxiv](https://img.shields.io/badge/CVPR-2025_Accepted-gold.svg?style=flat&logo=google-chrome)](【论文arxiv占位符】) -->

<!-- > ### 🎬 CineSRD
>
> A comprehensive framework for **speaker recognition and diarization** in cinematic content
>
> Combining **audio-visual fusion** with **LLM-based reasoning** to achieve speaker recognition and diarization in complex movie scenes.

📄 **Read the full paper:** [CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization](【占位符】) -->



> ### 🎬 CineSRD
>
> A **tri-modal framework** integrating **visual, acoustic, and linguistic cues** for robust speaker recognition and diarization in cinematic content.
>
> 🔹 **Audio-Visual Fusion**: Combines face tracking with voice embeddings.  
> 🔹 **LLM Reasoning**: Leverages linguistic context for precise speaker turn detection.  
> 🔹 **Open-World Ready**: Designed for the complex, dynamic nature of movies.

📄 **Read the full paper:** [CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization](【占位符】)


---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Citation](#citation)

---

## 🎯 Overview

CineSRD addresses the challenging task of speaker diarization in movies and TV shows, where multiple speakers often overlap and visual information is crucial for accurate identification. Our framework integrates:

- **Multi-modal Speaker Recognition**: Fuses audio embeddings with visual face recognition
- **Active Speaker Detection**: Identifies who is speaking using visual cues
- **Speaker Turn Detection**: Leverages LLMs to detect speaker changes from audio-text context
- **SubtitleSD Benchmark**: A comprehensive dataset for evaluating speaker diarization in cinematic content

---

## ✨ Features

### Speaker Detection

- **Audio-Visual Fusion**: Combines ERes2Net audio embeddings with face recognition
- **Multi-stage Pipeline**: 
  - Face detection and quality assessment
  - Active speaker detection
  - Audio-visual clustering and post-processing
  - Speaker role assignment via avatar matching
- **Robust to Overlaps**: Handles multiple simultaneous speakers
- **High Accuracy**: Optimized for cinematic content with complex scenes

### Speaker Turn Detection

- **LLM-Powered**: Uses Qwen2-Audio for audio-text based speaker change detection
- **Context-Aware**: Analyzes conversational context to identify speaker switches
- **Probabilistic Output**: Provides confidence scores for speaker turns


---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 12.0+ (for GPU acceleration)
- FFmpeg

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/CineSRD.git
cd CineSRD
```

2. **Install Speaker Detection dependencies**

```bash
cd code/speaker_detection
pip install -r requirements.txt
```

3. **Install Speaker Turn Detection dependencies**

```bash
cd ../speaker_turn_detection
pip install -r requirements.txt
```

4. **Download Pretrained Models**

Download the required pretrained models and place them in the appropriate directories:
- Audio embedding models (ERes2Net)
- Face detection models
- Speaker turn detection models (Qwen2-Audio)

---

## 🎬 Quick Start

### Speaker Detection

1. **Configure paths** in `code/speaker_detection/config.yaml`:


2. **Run the pipeline**:

```bash
cd code/speaker_detection
bash main.sh
```

### Speaker Turn Detection

1. **Configure** in `code/speaker_turn_detection/config.yaml`:


2. **Run inference**:

```bash
cd code/speaker_turn_detection
bash main.sh
```

---

## 📖 Usage

### Speaker Detection Pipeline

The speaker detection module provides a complete pipeline including:
- Video preprocessing and face detection
- Audio embedding extraction (ERes2Net)
- Active speaker detection
- Clustering and post-processing

### Speaker Turn Detection

Uses Qwen2-Audio to detect speaker changes from audio-text context.


---

## 📄 Citation

If you use CineSRD in your research, please cite:

```bibtex
【占位符】
```

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


---

## 🙏 Acknowledgments

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - For audio processing tools
- [ModelScope](https://github.com/modelscope/modelscope) - For model hub support
- [ms-swift](https://github.com/modelscope/swift) - For model training framework

---

## 📧 Contact

For questions or support, please open an issue on GitHub or contact us at [h13971032630@163.com](mailto:h13971032630@163.com).

