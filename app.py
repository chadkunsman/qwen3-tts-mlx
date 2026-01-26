"""Gradio web app for Qwen3-TTS on Apple Silicon."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

VOICES = ["Ryan", "Vivian", "Serena", "Aiden", "Dylan", "Eric", "Uncle_Fu", "Ono_Anna", "Sohee"]
INSTRUCT_PRESETS = [
    "excited and happy",
    "calm and soothing",
    "serious and professional",
    "warm and friendly",
]

PRESET_MODELS = {
    "1.7B-CustomVoice": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
}
CLONE_MODELS = {
    "0.6B-Base": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
}

OUTPUTS_DIR = Path("outputs")
SAVED_VOICES_DIR = Path("saved_voices")
OUTPUTS_DIR.mkdir(exist_ok=True)
SAVED_VOICES_DIR.mkdir(exist_ok=True)

models = {}


def get_model(model_path: str):
    """Lazy load models to save memory."""
    if model_path not in models:
        models[model_path] = load_model(model_path)
    return models[model_path]


def load_saved_voices() -> dict:
    """Scan saved_voices/ on startup and return dict of saved voices."""
    voices = {}
    for voice_dir in SAVED_VOICES_DIR.iterdir():
        if voice_dir.is_dir():
            audio_path = voice_dir / "audio.wav"
            transcript_path = voice_dir / "transcript.txt"
            metadata_path = voice_dir / "metadata.json"
            if audio_path.exists() and transcript_path.exists():
                voices[voice_dir.name] = {
                    "audio": str(audio_path),
                    "transcript": transcript_path.read_text().strip(),
                    "metadata": json.loads(metadata_path.read_text()) if metadata_path.exists() else {},
                }
    return voices


def save_cloned_voice(audio_path: str, transcript: str, name: str) -> str:
    """Persist a cloned voice to saved_voices/."""
    if not name.strip():
        raise gr.Error("Please enter a name for the voice")
    if not audio_path:
        raise gr.Error("Please upload reference audio first")
    if not transcript.strip():
        raise gr.Error("Please enter the transcript first")

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    voice_dir = SAVED_VOICES_DIR / safe_name

    if voice_dir.exists():
        raise gr.Error(f"Voice '{safe_name}' already exists")

    voice_dir.mkdir(parents=True)
    shutil.copy(audio_path, voice_dir / "audio.wav")
    (voice_dir / "transcript.txt").write_text(transcript.strip())
    (voice_dir / "metadata.json").write_text(json.dumps({
        "name": name.strip(),
        "created": datetime.now().isoformat(),
    }))

    return f"Voice '{safe_name}' saved"


def save_generation(audio: np.ndarray, voice: str, temp: float, instruct: str, is_clone: bool = False) -> dict:
    """Save generated audio to outputs/ and return metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "clone" if is_clone else voice
    filename = f"{prefix}_{timestamp}.wav"
    filepath = OUTPUTS_DIR / filename

    sf.write(filepath, audio, 24000)

    return {
        "path": str(filepath),
        "filename": filename,
        "voice": voice,
        "temperature": temp,
        "instruct": instruct,
        "timestamp": timestamp,
        "is_clone": is_clone,
    }


def rename_generation(history: list, index: int, new_name: str) -> list:
    """Rename a file on disk and update history."""
    if index < 0 or index >= len(history):
        return history

    entry = history[index]
    old_path = Path(entry["path"])

    if not new_name.endswith(".wav"):
        new_name = new_name + ".wav"

    new_path = old_path.parent / new_name

    if new_path.exists() and new_path != old_path:
        raise gr.Error(f"File '{new_name}' already exists")

    if old_path.exists():
        old_path.rename(new_path)

    history[index]["path"] = str(new_path)
    history[index]["filename"] = new_name

    return history


def delete_generation(history: list, index: int, delete_file: bool) -> list:
    """Remove entry from history and optionally delete file from disk."""
    if index < 0 or index >= len(history):
        return history

    entry = history[index]
    if delete_file:
        filepath = Path(entry["path"])
        if filepath.exists():
            filepath.unlink()

    return history[:index] + history[index + 1:]


def clear_history(history: list, delete_files: bool) -> list:
    """Clear all history and optionally delete all files."""
    if delete_files:
        for entry in history:
            filepath = Path(entry["path"])
            if filepath.exists():
                filepath.unlink()
    return []


def validate_temperature(temp_str: str) -> float:
    """Validate temperature input and return float."""
    try:
        temp = float(temp_str)
    except ValueError:
        raise gr.Error("Temperature must be a number")

    if temp < 0.0 or temp > 2.0:
        raise gr.Error("Temperature must be between 0.0 and 2.0")

    return temp


def generate_preset(
    text: str, voice: str, instruct: str, temp_str: str, model_name: str, history: list
) -> tuple:
    """Generate audio using preset voice."""
    if not text.strip():
        raise gr.Error("Please enter text to synthesize")

    temp = validate_temperature(temp_str)
    model_path = PRESET_MODELS[model_name]
    model = get_model(model_path)

    results = list(model.generate(
        text=text,
        voice=voice,
        instruct=instruct.strip() or None,
        temperature=temp,
    ))

    audio = np.array(results[0].audio)
    metadata = save_generation(audio, voice, temp, instruct.strip())

    new_history = [metadata] + history
    return metadata["path"], new_history


def generate_clone(
    text: str,
    saved_voice: str,
    ref_audio: str,
    ref_text: str,
    temp_str: str,
    model_name: str,
    history: list,
) -> tuple:
    """Generate audio using voice cloning."""
    if not text.strip():
        raise gr.Error("Please enter text to synthesize")

    saved_voices = load_saved_voices()

    if saved_voice and saved_voice != "-- Upload new --":
        if saved_voice not in saved_voices:
            raise gr.Error(f"Saved voice '{saved_voice}' not found")
        voice_data = saved_voices[saved_voice]
        actual_audio = voice_data["audio"]
        actual_text = voice_data["transcript"]
        voice_name = saved_voice
    else:
        if not ref_audio:
            raise gr.Error("Please upload a reference audio file")
        if not ref_text.strip():
            raise gr.Error("Please enter the reference audio transcript")
        actual_audio = ref_audio
        actual_text = ref_text.strip()
        voice_name = "custom"

    temp = validate_temperature(temp_str)
    model_path = CLONE_MODELS[model_name]
    model = get_model(model_path)

    results = list(model.generate(
        text=text,
        audio=actual_audio,
        ref_text=actual_text,
        temperature=temp,
    ))

    audio = np.array(results[0].audio)
    metadata = save_generation(audio, voice_name, temp, "", is_clone=True)

    new_history = [metadata] + history
    return metadata["path"], new_history


def get_saved_voice_choices():
    """Get list of saved voice names for dropdown."""
    voices = load_saved_voices()
    choices = ["-- Upload new --"] + list(voices.keys())
    return choices


def build_output_label(entry: dict) -> str:
    """Build label string for an audio entry."""
    voice_label = f"Clone: {entry['voice']}" if entry.get("is_clone") else entry["voice"]
    instruct_short = entry["instruct"][:20] + "..." if len(entry["instruct"]) > 20 else entry["instruct"]
    label = f"{entry['filename']} | {voice_label} | T={entry['temperature']}"
    if instruct_short:
        label += f" | {instruct_short}"
    return label


def refresh_history_display(history: list, skip_first: bool = False):
    """Refresh the history audio players."""
    updates = []
    start_idx = 1 if skip_first else 0
    for i in range(4):
        hist_idx = i + start_idx
        if hist_idx < len(history):
            entry = history[hist_idx]
            label = build_output_label(entry)
            updates.append(gr.Audio(value=entry["path"], label=label, visible=True))
        else:
            updates.append(gr.Audio(value=None, visible=False))
    return updates


CSS = """
.compact-input textarea { font-size: 14px !important; }
.compact-input input { font-size: 14px !important; }
.preset-btn { min-width: 0 !important; padding: 4px 8px !important; font-size: 12px !important; }
.history-section { border-left: 2px solid #444; padding-left: 16px; }
.generate-btn { margin-top: 8px !important; }
"""

with gr.Blocks(title="Qwen3-TTS") as app:
    history_state = gr.State([])

    gr.Markdown("# Qwen3-TTS")

    with gr.Row():
        # LEFT COLUMN - Controls
        with gr.Column(scale=1):
            with gr.Tabs():
                # PRESET VOICES TAB
                with gr.Tab("Preset Voices"):
                    preset_text = gr.Textbox(
                        label="Text",
                        placeholder="Enter text to synthesize...",
                        lines=4,
                        elem_classes=["compact-input"],
                    )

                    preset_voice = gr.Dropdown(
                        choices=VOICES,
                        value="Ryan",
                        label="Voice",
                    )

                    preset_model = gr.Dropdown(
                        choices=list(PRESET_MODELS.keys()),
                        value=list(PRESET_MODELS.keys())[0],
                        label="Model",
                    )

                    gr.Markdown("**Style presets**", elem_id="style-label")
                    with gr.Row():
                        preset_btns = []
                        for preset in INSTRUCT_PRESETS:
                            btn = gr.Button(preset, size="sm", elem_classes=["preset-btn"])
                            preset_btns.append(btn)

                    preset_instruct = gr.Textbox(
                        label="Style instruction",
                        placeholder="e.g., excited and happy...",
                        lines=1,
                        elem_classes=["compact-input"],
                    )

                    for btn, preset in zip(preset_btns, INSTRUCT_PRESETS):
                        btn.click(fn=lambda p=preset: p, outputs=preset_instruct)

                    preset_temp = gr.Textbox(
                        label="Temperature (0.0 - 2.0)",
                        value="1.0",
                        elem_classes=["compact-input"],
                    )

                    preset_btn = gr.Button("Generate", variant="primary", elem_classes=["generate-btn"])

                # VOICE CLONING TAB
                with gr.Tab("Voice Cloning"):
                    clone_text = gr.Textbox(
                        label="Text",
                        placeholder="Enter text to synthesize...",
                        lines=4,
                        elem_classes=["compact-input"],
                    )

                    saved_voice_dropdown = gr.Dropdown(
                        choices=get_saved_voice_choices(),
                        value="-- Upload new --",
                        label="Saved Voice",
                        interactive=True,
                    )

                    clone_ref_audio = gr.Audio(
                        label="Reference audio",
                        type="filepath",
                        sources=["upload"],
                    )

                    clone_ref_text = gr.Textbox(
                        label="Reference transcript",
                        placeholder="Words spoken in reference audio...",
                        lines=2,
                        elem_classes=["compact-input"],
                    )

                    with gr.Row():
                        save_voice_name = gr.Textbox(
                            label="Save as",
                            placeholder="Voice name...",
                            scale=2,
                            elem_classes=["compact-input"],
                        )
                        save_voice_btn = gr.Button("Save", size="sm", scale=1)

                    save_voice_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        visible=False,
                    )

                    clone_model = gr.Dropdown(
                        choices=list(CLONE_MODELS.keys()),
                        value=list(CLONE_MODELS.keys())[0],
                        label="Model",
                    )

                    clone_temp = gr.Textbox(
                        label="Temperature (0.0 - 2.0)",
                        value="1.0",
                        elem_classes=["compact-input"],
                    )

                    clone_btn = gr.Button("Generate", variant="primary", elem_classes=["generate-btn"])

        # RIGHT COLUMN - Output
        with gr.Column(scale=1, elem_classes=["history-section"]):
            gr.Markdown("### Output")

            # Main output - always visible, shows current/latest generation
            current_output = gr.Audio(
                label="Generated Audio",
                type="filepath",
            )

            # History - previous generations pushed down
            history_players = []
            for i in range(4):
                player = gr.Audio(
                    label=f"Previous {i + 1}",
                    type="filepath",
                    visible=False,
                )
                history_players.append(player)

            with gr.Accordion("Manage", open=False):
                with gr.Row():
                    clear_history_btn = gr.Button("Clear All", size="sm", variant="secondary")
                    delete_files_checkbox = gr.Checkbox(label="Delete files from disk", value=False)

                with gr.Row():
                    rename_index = gr.Number(label="Entry #", precision=0, minimum=1, scale=1)
                    rename_new_name = gr.Textbox(label="New filename", scale=2)
                    rename_btn = gr.Button("Rename", size="sm", scale=1)

                with gr.Row():
                    delete_index = gr.Number(label="Entry #", precision=0, minimum=1, scale=1)
                    delete_file_checkbox = gr.Checkbox(label="Delete file", value=False, scale=1)
                    delete_btn = gr.Button("Delete", size="sm", variant="stop", scale=1)

                refresh_voices_btn = gr.Button("Refresh saved voices", size="sm")

    # Event handlers
    def shift_history_preset(history):
        """Shift current to history before generating - shows existing history."""
        # Show current history items (they become the "previous" ones)
        history_updates = refresh_history_display(history, skip_first=False)
        return [gr.Audio(value=None, label="Generating...")] + history_updates

    def shift_history_clone(history):
        """Shift current to history before generating - shows existing history."""
        history_updates = refresh_history_display(history, skip_first=False)
        return [gr.Audio(value=None, label="Generating...")] + history_updates

    def do_generate_preset(text, voice, instruct, temp, model, history):
        path, new_history = generate_preset(text, voice, instruct, temp, model, history)
        current_label = build_output_label(new_history[0])
        return new_history, gr.Audio(value=path, label=current_label)

    def do_generate_clone(text, saved_voice, ref_audio, ref_text, temp, model, history):
        path, new_history = generate_clone(text, saved_voice, ref_audio, ref_text, temp, model, history)
        current_label = build_output_label(new_history[0])
        return new_history, gr.Audio(value=path, label=current_label)

    def do_clear_history(history, delete_files):
        new_history = clear_history(history, delete_files)
        current_update = gr.Audio(value=None, label="Generated Audio")
        history_updates = refresh_history_display(new_history, skip_first=True)
        return [new_history, current_update] + history_updates

    def do_rename(history, index, new_name):
        if not new_name.strip():
            raise gr.Error("Please enter a new filename")
        idx = int(index) - 1
        new_history = rename_generation(history, idx, new_name.strip())
        if len(new_history) > 0:
            current_label = build_output_label(new_history[0])
            current_update = gr.Audio(value=new_history[0]["path"], label=current_label)
        else:
            current_update = gr.Audio(value=None, label="Generated Audio")
        history_updates = refresh_history_display(new_history, skip_first=True)
        return [new_history, current_update] + history_updates

    def do_delete(history, index, delete_file):
        idx = int(index) - 1
        new_history = delete_generation(history, idx, delete_file)
        if len(new_history) > 0:
            current_label = build_output_label(new_history[0])
            current_update = gr.Audio(value=new_history[0]["path"], label=current_label)
        else:
            current_update = gr.Audio(value=None, label="Generated Audio")
        history_updates = refresh_history_display(new_history, skip_first=True)
        return [new_history, current_update] + history_updates

    def do_save_voice(audio, transcript, name):
        msg = save_cloned_voice(audio, transcript, name)
        new_choices = get_saved_voice_choices()
        return gr.update(value=msg, visible=True), gr.update(choices=new_choices)

    def do_refresh_voices():
        return gr.update(choices=get_saved_voice_choices())

    # Chain: first shift history (instant, updates history players), then generate (slow, only updates current)
    preset_btn.click(
        fn=shift_history_preset,
        inputs=[history_state],
        outputs=[current_output] + history_players,
    ).then(
        fn=do_generate_preset,
        inputs=[preset_text, preset_voice, preset_instruct, preset_temp, preset_model, history_state],
        outputs=[history_state, current_output],
    )

    clone_btn.click(
        fn=shift_history_clone,
        inputs=[history_state],
        outputs=[current_output] + history_players,
    ).then(
        fn=do_generate_clone,
        inputs=[clone_text, saved_voice_dropdown, clone_ref_audio, clone_ref_text, clone_temp, clone_model, history_state],
        outputs=[history_state, current_output],
    )

    clear_history_btn.click(
        fn=do_clear_history,
        inputs=[history_state, delete_files_checkbox],
        outputs=[history_state, current_output] + history_players,
    )

    rename_btn.click(
        fn=do_rename,
        inputs=[history_state, rename_index, rename_new_name],
        outputs=[history_state, current_output] + history_players,
    )

    delete_btn.click(
        fn=do_delete,
        inputs=[history_state, delete_index, delete_file_checkbox],
        outputs=[history_state, current_output] + history_players,
    )

    save_voice_btn.click(
        fn=do_save_voice,
        inputs=[clone_ref_audio, clone_ref_text, save_voice_name],
        outputs=[save_voice_status, saved_voice_dropdown],
    )

    refresh_voices_btn.click(
        fn=do_refresh_voices,
        outputs=saved_voice_dropdown,
    )

if __name__ == "__main__":
    app.launch(css=CSS)
