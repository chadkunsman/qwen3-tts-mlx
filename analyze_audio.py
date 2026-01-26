"""Analyze audio files for silence gaps - Phase 1 baseline analysis."""

import os
import wave
from pathlib import Path


def analyze_audio(filepath: str) -> dict:
    """Analyze audio file for silence issues."""
    with wave.open(filepath, "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        duration = frames / rate
        file_size = os.path.getsize(filepath)
        expected_size = frames * wav.getsampwidth() * wav.getnchannels()

    file_size_mb = file_size / (1024 * 1024)
    # 96 sec = max token limit hit, indicates silence padding
    is_broken = duration >= 95.0

    return {
        "file": Path(filepath).name,
        "duration_sec": round(duration, 2),
        "file_size_mb": round(file_size_mb, 2),
        "sample_rate": rate,
        "status": "BROKEN" if is_broken else "OK",
    }


def analyze_directory(dir_path: str) -> list[dict]:
    """Analyze all wav files in a directory."""
    results = []
    path = Path(dir_path)
    for wav_file in path.glob("*.wav"):
        result = analyze_audio(str(wav_file))
        # Extract temperature from filename
        name = wav_file.stem
        if name.startswith("temp_"):
            try:
                temp = float(name.replace("temp_", "").replace("_000", ""))
                result["temperature"] = temp
            except ValueError:
                result["temperature"] = None
        results.append(result)
    # Sort by temperature
    results.sort(key=lambda x: x.get("temperature", 999))
    return results


def print_results(results: list[dict]) -> None:
    """Print results in a table format."""
    print(f"{'Temp':<8} {'Duration':<12} {'Size (MB)':<12} {'Status':<10}")
    print("-" * 45)
    for r in results:
        temp = r.get("temperature", "N/A")
        print(f"{temp:<8} {r['duration_sec']:<12} {r['file_size_mb']:<12} {r['status']:<10}")


if __name__ == "__main__":
    print("=" * 60)
    print("TEMPERATURE SWEEP ANALYSIS (Ryan voice)")
    print("=" * 60)
    results = analyze_directory("voice_samples/ryan")
    print_results(results)

    # Count broken vs OK
    broken = sum(1 for r in results if r["status"] == "BROKEN")
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\nSummary: {ok} OK, {broken} BROKEN")
