#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chuyển Văn Bản Thành Giọng Nói - HuggingFace Spaces"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from typing import Any, Dict

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig

print("Đang tải model...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
print("Model đã tải!")

sampling_rate = model.sampling_rate

try:
    from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
    _ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)
except Exception:
    _ALL_LANGUAGES = ["Auto", "vi", "en", "zh", "ja", "ko", "fr", "de", "es"]

_LANG_DISPLAY = {
    "Auto": "Auto (Tự nhận diện)",
    "vi": "Tiếng Việt (Việt Nam)",
    "en": "English (Mỹ)",
    "zh": "中文 (Trung Quốc)",
    "ja": "日本語 (Nhật Bản)",
    "ko": "한국어 (Hàn Quốc)",
    "fr": "Français (Pháp)",
    "de": "Deutsch (Đức)",
    "es": "Español (Tây Ban Nha)",
    "th": "ไทย (Thái Lan)",
    "id": "Bahasa Indonesia",
    "ms": "Bahasa Melayu",
    "pt": "Português (Bồ Đào Nha)",
    "it": "Italiano (Ý)",
    "ru": "Русский (Nga)",
    "ar": "العربية (Ả Rập)",
    "hi": "हिन्दी (Hindi)",
    "tr": "Türkçe (Thổ Nhĩ Kỳ)",
}

_VI_LANGUAGES = list(_LANG_DISPLAY.keys())

_DISPLAY_TO_CODE = {}
for k, v in _LANG_DISPLAY.items():
    _DISPLAY_TO_CODE[v] = k

_VOICE_OPTIONS = [
    "Mai (Nữ) - Giọng chuẩn",
    "Nam (Nam) - Giọng chuẩn",
    "Linh (Nữ) - Giọng ấm",
    "Minh (Nam) - Giọng trầm",
    "Hà (Nữ) - Giọng trẻ",
    "Tùng (Nam) - Giọng đọc",
    "自动 (Tự động)",
]

_REGION_OPTIONS = [
    "Tự động",
    "Miền Bắc",
    "Miền Trung",
    "Miền Nam",
]

_STYLE_OPTIONS = [
    "Tự nhiên",
    "Trang trọng",
    "Trò chuyện",
    "Đọc truyện",
    "Thuyết minh",
    "Quảng cáo",
]

_PURPOSE_OPTIONS = [
    "Thuyết minh",
    "Podcast",
    "Video",
    "Game",
    "Giáo dục",
    "Quảng cáo",
    "Trò chuyện",
]

_PAUSE_OPTIONS = [
    "Tự động",
    "Ngắn",
    "Trung bình",
    "Dài",
    "Không ngắt",
]

def _gen_tts(text, lang, voice, region, style, purpose, speed, volume, pitch, pause):
    if not text or not text.strip():
        return None, "⚠️ Vui lòng nhập văn bản", 0

    language = _DISPLAY_TO_CODE.get(lang, None)
    if language == "Auto":
        language = None

    gen_config = OmniVoiceGenerationConfig(
        num_step=32,
        guidance_scale=2.0,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
    )

    instruct_parts = []
    if style and style != "Tự nhiên":
        style_map = {
            "Trang trọng": "formal, professional",
            "Trò chuyện": "casual, conversational",
            "Đọc truyện": "storytelling, expressive",
            "Thuyết minh": "narration, documentary style",
            "Quảng cáo": "promotional, energetic",
        }
        if style in style_map:
            instruct_parts.append(style_map[style])

    instruct = ", ".join(instruct_parts) if instruct_parts else None

    kw = dict(text=text.strip(), language=language, generation_config=gen_config)
    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if instruct:
        kw["instruct"] = instruct

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"❌ Lỗi: {type(e).__name__}: {e}", 0

    vol = float(volume) if volume else 0.8
    waveform = (audio[0] * 32767 * vol).astype(np.int16)
    duration = len(waveform) / sampling_rate
    return (sampling_rate, waveform), f"✅ Hoàn tất! Thời lượng: {duration:.1f}s", duration

CUSTOM_CSS = """
* {box-sizing: border-box !important;}
body {background: #f8f9fc !important; font-family: 'Segoe UI', system-ui, sans-serif !important;}

.main-wrap {max-width: 900px !important; margin: 0 auto !important; padding: 0 !important;}
.gradio-container {background: transparent !important; box-shadow: none !important; border-radius: 0 !important; max-width: 100% !important;}

/* === HEADER === */
.hero-header {background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    border-radius: 20px; padding: 35px 40px 30px; margin-bottom: 0; position: relative; overflow: hidden;}
.hero-header::after {content: ''; position: absolute; right: 40px; top: 50%; transform: translateY(-50%);
    width: 180px; height: 180px; background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    border-radius: 50%;}
.hero-title {font-size: 2em !important; font-weight: 800 !important; color: white !important; margin: 0 0 6px 0 !important;}
.hero-sub {color: rgba(255,255,255,0.85) !important; font-size: 0.95em !important; margin: 0 0 24px 0 !important;}

/* === FEATURES === */
.features-row {display: flex !important; gap: 16px !important; flex-wrap: wrap !important;}
.feature-item {flex: 1 !important; min-width: 120px !important; text-align: center !important;
    background: rgba(255,255,255,0.12) !important; border-radius: 12px !important; padding: 14px 8px !important;
    backdrop-filter: blur(10px) !important;}
.feature-icon {width: 40px !important; height: 40px !important; border-radius: 10px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    margin: 0 auto 8px !important; font-size: 1.2em !important;}
.feature-icon.yellow {background: #fef3c7 !important;}
.feature-icon.blue {background: #dbeafe !important;}
.feature-icon.green {background: #d1fae5 !important;}
.feature-icon.purple {background: #ede9fe !important;}
.feature-label {font-size: 0.85em !important; font-weight: 700 !important; color: white !important; margin: 0 0 2px 0 !important;}
.feature-desc {font-size: 0.72em !important; color: rgba(255,255,255,0.7) !important; margin: 0 !important;}

/* === STEPS === */
.steps-bar {display: flex !important; align-items: center !important; justify-content: center !important;
    background: white !important; border-radius: 16px !important; padding: 20px 30px !important;
    margin: 20px 0 !important; box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;}
.step-item {display: flex !important; align-items: center !important; gap: 10px !important;}
.step-circle {width: 36px !important; height: 36px !important; border-radius: 50% !important;
    background: #6366f1 !important; color: white !important; display: flex !important;
    align-items: center !important; justify-content: center !important;
    font-weight: 700 !important; font-size: 0.9em !important; flex-shrink: 0 !important;}
.step-circle.inactive {background: #e5e7eb !important; color: #9ca3af !important;}
.step-info {text-align: left !important;}
.step-title {font-size: 0.9em !important; font-weight: 700 !important; color: #1f2937 !important; margin: 0 !important;}
.step-desc {font-size: 0.75em !important; color: #9ca3af !important; margin: 0 !important;}
.step-dots {flex: 1 !important; border-top: 2px dashed #d1d5db !important; margin: 0 16px !important; min-width: 40px !important;}

/* === CARDS === */
.card {background: white !important; border-radius: 16px !important; padding: 28px 30px !important;
    margin-bottom: 16px !important; box-shadow: 0 2px 12px rgba(0,0,0,0.05) !important;
    border: 1px solid #f0f0f5 !important;}
.card-header {display: flex !important; align-items: center !important; gap: 10px !important; margin-bottom: 16px !important;}
.card-icon {width: 32px !important; height: 32px !important; border-radius: 8px !important;
    background: #ede9fe !important; display: flex !important; align-items: center !important;
    justify-content: center !important; font-size: 1em !important; flex-shrink: 0 !important;}
.card-title {font-size: 1.1em !important; font-weight: 700 !important; color: #1f2937 !important; margin: 0 !important;}
.card-subtitle {font-size: 0.8em !important; color: #9ca3af !important; margin: 2px 0 0 0 !important;}

/* === TEXT INPUT === */
.text-area-wrap {position: relative !important;}
.text-counter {position: absolute !important; top: 12px !important; right: 12px !important;
    font-size: 0.8em !important; color: #9ca3af !important;}
.text-actions {display: flex !important; gap: 8px !important; margin-top: 10px !important; flex-wrap: wrap !important;}
.text-action-btn {background: #f3f4f6 !important; border: 1px solid #e5e7eb !important;
    border-radius: 8px !important; padding: 8px 16px !important; font-size: 0.82em !important;
    color: #6b7280 !important; cursor: pointer !important; display: flex !important;
    align-items: center !important; gap: 6px !important; transition: all 0.2s !important;}
.text-action-btn:hover {background: #e5e7eb !important;}
.text-action-btn.danger {color: #ef4444 !important; border-color: #fecaca !important;}
.text-action-btn.danger:hover {background: #fef2f2 !important;}

/* === SETTINGS GRID === */
.settings-grid {display: grid !important; grid-template-columns: 1fr 1fr !important; gap: 14px !important;}
.setting-box {background: #f9fafb !important; border: 1px solid #e5e7eb !important;
    border-radius: 12px !important; padding: 14px 16px !important;}
.setting-label {font-size: 0.82em !important; font-weight: 600 !important; color: #6b7280 !important;
    margin-bottom: 6px !important; display: flex !important; align-items: center !important; gap: 6px !important;}
.setting-full {grid-column: 1 / -1 !important;}

/* === SLIDERS === */
.slider-row {display: flex !important; align-items: center !important; gap: 12px !important;}
.slider-row input[type="range"] {flex: 1 !important;}
.slider-val {font-size: 0.9em !important; font-weight: 600 !important; color: #6366f1 !important;
    min-width: 45px !important; text-align: right !important;}

/* === GENERATE BUTTON === */
.gen-btn-wrap {display: flex !important; gap: 12px !important; align-items: stretch !important; margin: 20px 0 10px 0 !important;}
.gen-btn {flex: 1 !important; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important; border: none !important; border-radius: 16px !important;
    padding: 18px 30px !important; font-size: 1.15em !important; font-weight: 700 !important;
    cursor: pointer !important; box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    transition: all 0.3s !important; display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;}
.gen-btn:hover {transform: translateY(-2px) !important; box-shadow: 0 6px 25px rgba(99,102,241,0.5) !important;}
.gen-btn-sub {font-size: 0.7em !important; font-weight: 400 !important; opacity: 0.85 !important; margin-top: 4px !important;}
.history-btn {background: white !important; border: 1px solid #e5e7eb !important; border-radius: 14px !important;
    padding: 14px 20px !important; cursor: pointer !important; display: flex !important;
    flex-direction: column !important; align-items: center !important; justify-content: center !important;
    gap: 4px !important; color: #6b7280 !important; font-size: 0.8em !important; min-width: 70px !important;}
.history-btn:hover {background: #f9fafb !important;}

.privacy-note {text-align: center !important; font-size: 0.78em !important; color: #9ca3af !important;
    margin: 8px 0 20px 0 !important;}

/* === RESULT === */
.result-tabs {display: flex !important; gap: 8px !important; margin-bottom: 16px !important;}
.result-tab {padding: 8px 18px !important; border-radius: 8px !important; font-size: 0.85em !important;
    font-weight: 600 !important; cursor: pointer !important; border: 1px solid #e5e7eb !important;
    background: white !important; color: #6b7280 !important; display: flex !important;
    align-items: center !important; gap: 6px !important;}
.result-tab.active {background: #6366f1 !important; color: white !important; border-color: #6366f1 !important;}

.audio-player-wrap {background: #f9fafb !important; border-radius: 12px !important; padding: 20px !important;
    text-align: center !important;}
.download-row {display: flex !important; justify-content: center !important; gap: 16px !important;
    margin-top: 16px !important; flex-wrap: wrap !important;}
.dl-btn {display: flex !important; align-items: center !important; gap: 6px !important;
    font-size: 0.85em !important; color: #6366f1 !important; cursor: pointer !important;
    text-decoration: none !important; font-weight: 600 !important;}
.dl-btn:hover {text-decoration: underline !important;}

.tip-note {text-align: center !important; font-size: 0.78em !important; color: #9ca3af !important;
    margin-top: 20px !important; padding: 10px !important;}

/* === OVERRIDE GRADIO === */
.gradio-container .prose {display: none !important;}
section {border: none !important; padding: 0 !important; margin: 0 !important;}
.gr-group {margin: 0 !important; padding: 0 !important;}
"""

with gr.Blocks(css=CUSTOM_CSS, title="Chuyển Văn Bản Thành Giọng Nói") as demo:

    # ===================== HEADER =====================
    gr.HTML("""
    <div class="hero-header">
        <div class="hero-title">Chuyển Văn Bản Thành Giọng Nói</div>
        <div class="hero-sub">AI chuyển văn bản thành giọng nói tự nhiên, chân thực như người thật</div>
        <div class="features-row">
            <div class="feature-item">
                <div class="feature-icon yellow">⚡</div>
                <div class="feature-label">Giọng nói tự nhiên</div>
                <div class="feature-desc">Như người thật</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon blue">🧠</div>
                <div class="feature-label">AI thông minh</div>
                <div class="feature-desc">Xử lý nhanh chóng</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon green">🌐</div>
                <div class="feature-label">Đa ngôn ngữ</div>
                <div class="feature-desc">Hỗ trợ 100+ ngôn ngữ</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon purple">🛡️</div>
                <div class="feature-label">Bảo mật tuyệt đối</div>
                <div class="feature-desc">Tải về hoặc chia sẻ</div>
            </div>
        </div>
    </div>
    """)

    # ===================== STEPS =====================
    gr.HTML("""
    <div class="steps-bar">
        <div class="step-item">
            <div class="step-circle">1</div>
            <div class="step-info">
                <div class="step-title">Nhập văn bản</div>
                <div class="step-desc">Hoặc dán nội dung</div>
            </div>
        </div>
        <div class="step-dots"></div>
        <div class="step-item">
            <div class="step-circle inactive">2</div>
            <div class="step-info">
                <div class="step-title">Chọn giọng nói</div>
                <div class="step-desc">Thiết lập giọng đọc</div>
            </div>
        </div>
        <div class="step-dots"></div>
        <div class="step-item">
            <div class="step-circle inactive">3</div>
            <div class="step-info">
                <div class="step-title">Tạo giọng nói</div>
                <div class="step-desc">Tải về hoặc chia sẻ</div>
            </div>
        </div>
    </div>
    """)

    # ===================== TEXT INPUT =====================
    gr.HTML("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">📝</div>
            <div>
                <div class="card-title">Nhập văn bản của bạn</div>
                <div class="card-subtitle">Nhập hoặc dán nội dung bạn muốn chuyển thành giọng nói</div>
            </div>
        </div>
    """)

    input_text = gr.Textbox(
        label="", lines=6, show_label=False,
        placeholder="Nhập văn bản hoặc dán nội dung tại đây...\n\nGợi ý mẫu:\n• Xin chào, hôm nay thời tiết thật đẹp!\n• Trí tuệ nhân tạo đang thay đổi thế giới.\n• Thành công là kết quả của sự kiên trì mỗi ngày.",
        elem_classes="text-area-wrap",
    )

    gr.HTML("""
        <div class="text-actions">
            <span class="text-action-btn">📋 Dán từ clipboard</span>
            <span class="text-action-btn">📄 Tải file (.txt, .docx)</span>
            <span class="text-action-btn danger" id="clear-btn">🗑️ Xóa nội dung</span>
        </div>
    </div>
    """)

    # ===================== LANGUAGE & VOICE =====================
    gr.HTML("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">🗣️</div>
            <div>
                <div class="card-title">Ngôn ngữ & Giọng nói</div>
                <div class="card-subtitle">Chọn ngôn ngữ và giọng đọc phù hợp</div>
            </div>
        </div>
        <div class="settings-grid">
    """)

    with gr.Row():
        lang_dd = gr.Dropdown(
            label="🌐 Ngôn ngữ",
            choices=_VI_LANGUAGES,
            value="vi",
            elem_classes="setting-box",
        )
        voice_dd = gr.Dropdown(
            label="🎙️ Giọng nói",
            choices=_VOICE_OPTIONS,
            value="Mai (Nữ) - Giọng chuẩn",
            elem_classes="setting-box",
        )

    with gr.Row():
        region_dd = gr.Dropdown(
            label="📍 Vùng miền",
            choices=_REGION_OPTIONS,
            value="Tự động",
            elem_classes="setting-box",
        )
        style_dd = gr.Dropdown(
            label="🎨 Phong cách",
            choices=_STYLE_OPTIONS,
            value="Tự nhiên",
            elem_classes="setting-box",
        )
        purpose_dd = gr.Dropdown(
            label="📌 Mục đích sử dụng",
            choices=_PURPOSE_OPTIONS,
            value="Thuyết minh",
            elem_classes="setting-box",
        )

    gr.HTML("</div></div>")

    # ===================== ADVANCED SETTINGS =====================
    gr.HTML("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">⚙️</div>
            <div>
                <div class="card-title">Tùy chỉnh nâng cao</div>
                <div class="card-subtitle">Điều chỉnh âm lượng, tốc độ và các thiết lập khác</div>
            </div>
        </div>
    """)

    with gr.Row():
        speed_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Tốc độ đọc")
        volume_slider = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Âm lượng")

    with gr.Row():
        pitch_slider = gr.Slider(-20, 20, value=0, step=1, label="Cao độ (Pitch)")
        pause_dd = gr.Dropdown(label="Ngắt nghỉ", choices=_PAUSE_OPTIONS, value="Tự động")

    with gr.Accordion("⚙️ Tùy chọn nâng cao khác", open=False):
        with gr.Row():
            num_step_slider = gr.Slider(4, 64, value=32, step=1, label="Bước suy luận")
            cfg_slider = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="CFG Scale")

    gr.HTML("</div>")

    # ===================== GENERATE BUTTON =====================
    gr.HTML("""
    <div class="gen-btn-wrap">
    """)
    gen_btn = gr.Button("🔊 TẠO GIỌNG NÓI", elem_classes="gen-btn")
    gr.HTML("""
        <button class="history-btn" disabled>
            <span style="font-size:1.4em;">🕐</span>
            <span>Lịch sử</span>
        </button>
    </div>
    <div class="privacy-note">🔒 Dữ liệu của bạn được bảo mật và sẽ không được lưu trữ</div>
    """)

    # ===================== RESULT =====================
    gr.HTML("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">🔊</div>
            <div>
                <div class="card-title">Kết quả</div>
                <div class="card-subtitle">Nghe và tải về file giọng nói</div>
            </div>
        </div>
        <div class="result-tabs">
            <span class="result-tab active">🎵 Audio</span>
            <span class="result-tab">📝 Văn bản</span>
        </div>
    """)

    output_audio = gr.Audio(label="", type="numpy")
    output_status = gr.Textbox(label="", lines=1, interactive=False, show_label=False,
        placeholder="Trạng thái sẽ hiển thị ở đây...")

    gr.HTML("""
        <div class="download-row">
            <span class="dl-btn">📥 Tải xuống MP3</span>
            <span class="dl-btn">📥 Tải xuống WAV</span>
            <span class="dl-btn">🔗 Chia sẻ</span>
        </div>
    </div>
    """)

    gr.HTML("""
    <div class="tip-note">
        💡 Mẹo: Sử dụng dấu câu để tạo ngắt nghỉ tự nhiên. Ví dụ: dấu phẩy (,) ngắt ngắn, dấu chấm (.) ngắt dài.
    </div>
    """)

    # ===================== HANDLER =====================
    def _generate(text, lang, voice, region, style, purpose, speed, vol, pitch, pause):
        return _gen_tts(text, lang, voice, region, style, purpose, speed, vol, pitch, pause)

    gen_btn.click(
        _generate,
        inputs=[input_text, lang_dd, voice_dd, region_dd, style_dd, purpose_dd,
                speed_slider, volume_slider, pitch_slider, pause_dd],
        outputs=[output_audio, output_status],
        api_name="tts",
    )

if __name__ == "__main__":
    demo.queue().launch()
