# Qwen3-TTS MLX

A Gradio web app for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech on Apple Silicon, powered by [mlx-audio](https://github.com/Blaizzy/mlx-audio).

## Features

- **Preset voices** - 9 built-in voices with style instruction control
- **Voice design** - Create custom voices from text descriptions
- **Voice cloning** - Clone voices from short reference audio clips (M4A, MP3, WAV)
- **Model management** - Download and delete models from the UI
- **Generation history** - 5 recent outputs with waveform playback, inline rename, and delete

## Requirements

- macOS with Apple Silicon (M1+)
- Python 3.11+
- [UV](https://docs.astral.sh/uv/) for environment management

## Setup

```bash
git clone https://github.com/chadkunsman/qwen3-tts-mlx.git
cd qwen3-tts-mlx
uv sync
```

## Usage

```bash
uv run python app.py
# Opens at http://127.0.0.1:7860
```

On first use, go to the **Models** tab and download at least one model. The 1.7B-CustomVoice model is recommended for preset voices.

## Models

| Model | Size | Use Case |
|-------|------|----------|
| 1.7B-CustomVoice | ~4.5 GB | Preset voices with style instructions |
| 1.7B-VoiceDesign | ~4.5 GB | Custom voice creation from descriptions |
| 1.7B-Base | ~4.5 GB | Voice cloning (higher quality) |
| 0.6B-Base | ~1.5 GB | Voice cloning (faster, smaller) |

Models are downloaded to `~/.cache/huggingface/hub/` and can be managed from the Models tab.

## Available Voices

| Voice | Description | Best For |
|-------|-------------|----------|
| Ryan | Dynamic, strong rhythm | English |
| Aiden | Sunny, clear midrange | English |
| Vivian | Bright, slightly edgy | Chinese |
| Serena | Warm, gentle | Chinese |
| Dylan | Youthful Beijing | Chinese (Beijing) |
| Eric | Lively, husky | Chinese (Sichuan) |
| Uncle_Fu | Seasoned, low mellow | Chinese |
| Ono_Anna | Playful | Japanese |
| Sohee | Warm | Korean |

## Voice Cloning

1. Go to the **Clone Voice** tab and upload a reference audio clip with its transcript
2. Save the voice with a name
3. Switch to the **Voice Cloning** tab to generate speech using the saved voice

Saved voices are stored in `saved_voices/`.

## Supported Languages

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
