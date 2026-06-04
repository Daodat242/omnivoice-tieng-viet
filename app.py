#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OmniVoice - Chuyển văn bản thành giọng nói (HuggingFace Spaces)"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import gradio as gr
import torch
import numpy as np
import soundfile as sf
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

# Tải model
print("Đang tải model OmniVoice...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
print("Model đã tải!")

# Ánh xạ giọng tiếng Việt sang instruct
VOICE_MAP = {
    "Tự động": None,
    "Nam - Trung niên": "male, middle-aged, moderate pitch",
    "Nam - Trẻ tuổi": "male, young adult, moderate pitch",
    "Nam - Già": "male, elderly, low pitch",
    "Nam - Thiếu niên": "male, teenager, high pitch",
    "Nữ - Trung niên": "female, middle-aged, moderate pitch",
    "Nữ - Trẻ tuổi": "female, young adult, moderate pitch",
    "Nữ - Già": "female, elderly, low pitch",
    "Nữ - Thiếu niên": "female, teenager, high pitch",
    "Trẻ em": "child, moderate pitch",
    "Thì thầm (Nam)": "male, whisper",
    "Thì thầm (Nữ)": "female, whisper",
}

def generate_speech(text, voice_type, speed, num_steps):
    """Chuyển văn bản thành giọng nói"""
    if not text or not text.strip():
        return None, "Vui lòng nhập văn bản!"

    # Lấy instruct từ voice map
    instruct = VOICE_MAP.get(voice_type)

    # Tạo generation config
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
        return (24000, waveform), "Hoàn tất!"
    except Exception as e:
        return None, f"Lỗi: {str(e)}"

# Tạo giao diện
with gr.Blocks(title="OmniVoice - Tiếng Việt") as demo:
    gr.Markdown("""
# OmniVoice - Chuyển văn bản thành giọng nói

Mô hình AI hỗ trợ **hơn 600 ngôn ngữ**, bao gồm **tiếng Việt**.

**Hướng dẫn:**
1. Nhập văn bản tiếng Việt
2. Chọn giọng nói
3. Nhấn "Tạo giọng" và chờ kết quả
""")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Nhập văn bản tiếng Việt",
                lines=5,
                placeholder="VD: Xin chào các bạn. Hôm nay chúng ta học về AI.",
            )

            voice_dropdown = gr.Dropdown(
                label="Chọn giọng nói",
                choices=list(VOICE_MAP.keys()),
                value="Tự động",
            )

            with gr.Row():
                speed_slider = gr.Slider(
                    0.5, 1.5, value=1.0, step=0.1,
                    label="Tốc độ",
                )
                steps_slider = gr.Slider(
                    4, 64, value=32, step=4,
                    label="Số bước (thấp = nhanh)",
                )

            generate_btn = gr.Button("Tạo giọng", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Kết quả", type="numpy")
            status_output = gr.Textbox(label="Trạng thái", interactive=False)

    generate_btn.click(
        generate_speech,
        inputs=[text_input, voice_dropdown, speed_slider, steps_slider],
        outputs=[audio_output, status_output],
    )

    gr.Markdown("""
---
**Lưu ý:**
- Văn bản ngắn (< 100 từ) sẽ chạy nhanh hơn
- Số bước thấp (4-16) = nhanh hơn, chất lượng kém hơn
- Số bước cao (32-64) = chậm hơn, chất lượng tốt hơn
-由 [OmniVoice](https://github.com/k2-fsa/OmniVoice) tạo bởi Xiaomi AI Lab
""")

if __name__ == "__main__":
    demo.queue().launch()
