#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OmniVoice - Chuyen van ban thanh giong noi (HuggingFace Spaces)"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import gradio as gr
import torch
import numpy as np
import soundfile as sf
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

# Tai model
print("Dang tai model OmniVoice...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
print("Model da tai!")

# Map giong tieng Viet sang instruct
VOICE_MAP = {
    "Tu dong": None,
    "Nam - Trung nien": "male, middle-aged, moderate pitch",
    "Nam - Tre tuoi": "male, young adult, moderate pitch",
    "Nam - Gia": "male, elderly, low pitch",
    "Nam - Thieu nien": "male, teenager, high pitch",
    "Nu - Trung nien": "female, middle-aged, moderate pitch",
    "Nu - Tre tuoi": "female, young adult, moderate pitch",
    "Nu - Gia": "female, elderly, low pitch",
    "Nu - Thieu nien": "female, teenager, high pitch",
    "Tre em": "child, moderate pitch",
    "Thi tham (Nam)": "male, whisper",
    "Thi tham (Nu)": "female, whisper",
}

def generate_speech(text, voice_type, speed, num_steps):
    """Chuyen van ban thanh giong noi"""
    if not text or not text.strip():
        return None, "Vui long nhap van ban!"

    # Lay instruct tu voice map
    instruct = VOICE_MAP.get(voice_type)

    # Tao generation config
    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_steps),
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
    )

    # Generate
    kw = dict(
        text=text.strip(),
        language="vi",
        generation_config=gen_config,
    )

    if instruct:
        kw["instruct"] = instruct
    if speed != 1.0:
        kw["speed"] = speed

    try:
        audio = model.generate(**kw)
        waveform = (audio[0] * 32767).astype(np.int16)
        return (24000, waveform), "Hoan tat!"
    except Exception as e:
        return None, f"Loi: {str(e)}"

# Tao giao dien
with gr.Blocks(title="OmniVoice - Tieng Viet") as demo:
    gr.Markdown("""
# OmniVoice - Chuyen van ban thanh giong noi

Mô hình AI ho tro **600+ ngon ngu**, bao gom **tieng Viet**.

**Huong dan:**
1. Nhap van ban tieng Viet
2. Chon giong noi
3. Nhan "Tao giong" va cho ket qua
""")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Nhap van ban tieng Viet",
                lines=5,
                placeholder="VD: Xin chao cac ban. Hom nay chung ta hoc ve AI.",
            )

            voice_dropdown = gr.Dropdown(
                label="Chon giong noi",
                choices=list(VOICE_MAP.keys()),
                value="Tu dong",
            )

            with gr.Row():
                speed_slider = gr.Slider(
                    0.5, 1.5, value=1.0, step=0.1,
                    label="Toc do",
                )
                steps_slider = gr.Slider(
                    4, 64, value=32, step=4,
                    label="So buoc (thap = nhanh)",
                )

            generate_btn = gr.Button("Tao giong", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Ket qua", type="numpy")
            status_output = gr.Textbox(label="Trang thai", interactive=False)

    generate_btn.click(
        generate_speech,
        inputs=[text_input, voice_dropdown, speed_slider, steps_slider],
        outputs=[audio_output, status_output],
    )

    gr.Markdown("""
---
**Luu y:**
- Van ban ngan (< 100 tu) se chay nhanh hon
- So buoc thap (4-16) = nhanh hon, chat luong kem hon
- So buoc cao (32-64) = cham hon, chat luong tot hon
- Tai [OmniVoice](https://github.com/k2-fsa/OmniVoice) boi Xiaomi AI Lab
""")

if __name__ == "__main__":
    demo.queue().launch()
