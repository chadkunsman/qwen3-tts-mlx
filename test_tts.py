"""Quick test script for Qwen3-TTS via MLX."""

import subprocess
import sys

# Test texts to evaluate quality
TESTS = [
    "Hello, this is a test of the Qwen3 text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell.",
]

MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"


def generate(text: str, output: str):
    """Generate audio from text."""
    cmd = [
        sys.executable,
        "-m",
        "mlx_audio.tts.generate",
        "--model",
        MODEL,
        "--text",
        text,
        "--output",
        output,
    ]
    print(f"Generating: {output}")
    subprocess.run(cmd, check=True)
    print(f"Done: {output}\n")


if __name__ == "__main__":
    for i, text in enumerate(TESTS, 1):
        generate(text, f"test_{i}.wav")

    print("All tests complete! Check test_1.wav, test_2.wav, test_3.wav")
