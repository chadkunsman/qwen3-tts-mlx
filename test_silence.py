"""Automated test runner for silence gap investigation."""

import subprocess
import sys
from pathlib import Path

# Toggle between models for testing
MODEL_0_6B = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
MODEL_1_7B = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"

# Current test configuration
MODEL = MODEL_1_7B
VOICE = "Ryan"  # 1.7B-CustomVoice uses named voices
OUTPUT_DIR = Path("voice_samples/1_7B_high_temp")  # High temp tests

TEST_TEXT = """Okay, we're about to try something new. I'm really excited to try this out, and I hope it works out well for all of us here. Thank you so much for trying this out, I appreciate it so much."""


def generate(text: str, temperature: float, output_prefix: str) -> Path:
    """Generate audio at specified temperature."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "mlx_audio.tts.generate",
        "--model",
        MODEL,
        "--text",
        text,
        "--temperature",
        str(temperature),
        "--output_path",
        str(OUTPUT_DIR),
        "--file_prefix",
        output_prefix,
    ]
    if VOICE:
        cmd.extend(["--voice", VOICE])
    print(f"Generating: temp={temperature}")
    subprocess.run(cmd, check=True)
    return OUTPUT_DIR / f"{output_prefix}_000.wav"


def run_temperature_sweep(temperatures: list[float]) -> None:
    """Run tests across multiple temperatures."""
    for temp in temperatures:
        prefix = f"temp_{temp}"
        output_file = OUTPUT_DIR / f"{prefix}_000.wav"
        if output_file.exists():
            print(f"Skipping temp={temp} (already exists)")
            continue
        try:
            generate(TEST_TEXT, temp, prefix)
        except subprocess.CalledProcessError as e:
            print(f"FAILED at temp={temp}: {e}")


def run_consistency_test(temperatures: list[float], trials: int = 3) -> None:
    """Run multiple trials at each temperature to test consistency."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for temp in temperatures:
        for trial in range(1, trials + 1):
            prefix = f"consistency_t{temp}_r{trial}"
            output_file = OUTPUT_DIR / f"{prefix}_000.wav"
            if output_file.exists():
                print(f"Skipping temp={temp} trial={trial} (already exists)")
                continue
            try:
                generate(TEST_TEXT, temp, prefix)
            except subprocess.CalledProcessError as e:
                print(f"FAILED at temp={temp} trial={trial}: {e}")


if __name__ == "__main__":
    # High temperature test: 1.0 to 1.5
    temperatures = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    run_consistency_test(temperatures, trials=1)  # Single trial per temp
