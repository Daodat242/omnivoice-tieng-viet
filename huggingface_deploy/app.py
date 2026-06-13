#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chuyển Văn Bản Thành Giọng Nói - HuggingFace Spaces"""

import sys, io, hashlib
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

_LANG_DISPLAY = {
    "Auto": "Auto (Tự nhận diện)", "vi": "Tiếng Việt (Việt Nam)",
    "en": "English (Mỹ)", "zh": "中文 (Trung Quốc)", "ja": "日本語 (Nhật Bản)",
    "ko": "한국어 (Hàn Quốc)", "fr": "Français (Pháp)", "de": "Deutsch (Đức)",
    "es": "Español (Tây Ban Nha)", "th": "ไทย (Thái Lan)", "id": "Bahasa Indonesia",
    "pt": "Português (Bồ Đào Nha)", "it": "Italiano (Ý)", "ru": "Русский (Nga)",
    "ar": "العربية (Ả Rập)", "hi": "हिन्दी (Hindi)", "tr": "Türkçe (Thổ Nhĩ Kỳ)",
}
_VI_LANGUAGES = list(_LANG_DISPLAY.keys())
_DISPLAY_TO_CODE = {v: k for k, v in _LANG_DISPLAY.items()}

_VOICE_OPTIONS = ["Mai (Nữ) - Giọng chuẩn", "Nam (Nam) - Giọng chuẩn", "Linh (Nữ) - Giọng ấm",
    "Minh (Nam) - Giọng trầm", "Hà (Nữ) - Giọng trẻ", "Tùng (Nam) - Giọng đọc", "Tự động"]
_STYLE_OPTIONS = ["Tự nhiên", "Trang trọng", "Trò chuyện", "Đọc truyện", "Thuyết minh", "Quảng cáo"]
_PAUSE_OPTIONS = ["Tự động", "Ngắn", "Trung bình", "Dài", "Không ngắt"]
_ACCENT_PRESETS = {
    "Không chọn": "", "Giọng Mỹ": "american accent", "Giọng Anh": "british accent",
    "Giọng Úc": "australian accent", "Giọng Ấn Độ": "indian accent",
    "Giọng Nhật Bản": "japanese accent", "Giọng Hàn Quốc": "korean accent",
}
_DESIGN_PRESETS = {
    "Không chọn": "", "Nam - Trung niên": "male, middle-aged, moderate pitch",
    "Nam - Trẻ tuổi": "male, young adult, moderate pitch", "Nam - Già": "male, elderly, low pitch",
    "Nữ - Trung niên": "female, middle-aged, moderate pitch", "Nữ - Trẻ tuổi": "female, young adult, moderate pitch",
    "Nữ - Già": "female, elderly, low pitch", "Trẻ em": "child, moderate pitch",
}
_PURPOSE_MAP = {
    "TTS cho ứng dụng di động": "mobile_tts", "TTS cho website": "web_tts",
    "Clone giọng cho chatbot": "chatbot_clone", "Thiết kế giọng cho video": "video_design",
    "Tích hợp hệ thống giáo dục": "education", "Sử dụng cá nhân": "personal",
}

def generate_api_key(name, purpose):
    purpose_code = _PURPOSE_MAP.get(purpose, "general")
    raw = f"{name.strip().lower()}_{purpose_code}_omnivoice_secret_2024"
    return f"OV-{purpose_code.upper()[:8]}-{hashlib.sha256(raw.encode()).hexdigest()[:16].upper()}"

def _gen_core(text, language, ref_audio, instruct, num_step, guidance_scale,
              denoise, speed, duration, preprocess_prompt, postprocess_output, ref_text=None):
    if not text or not text.strip():
        return None, "⚠️ Vui lòng nhập văn bản"
    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32), guidance_scale=float(guidance_scale or 2.0),
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt) if preprocess_prompt is not None else True,
        postprocess_output=bool(postprocess_output) if postprocess_output is not None else True,
    )
    lang = language if (language and language != "Auto") else None
    kw: Dict[str, Any] = dict(text=text.strip(), language=lang, generation_config=gen_config)
    if speed is not None and float(speed) != 1.0: kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0: kw["duration"] = float(duration)
    if ref_audio: kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    if instruct and instruct.strip(): kw["instruct"] = instruct.strip()
    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"❌ Lỗi: {type(e).__name__}: {e}"
    waveform = (audio[0] * 32767).astype(np.int16)
    return (sampling_rate, waveform), "✅ Hoàn tất!"

def _gen_tts(text, lang, voice, style, speed, volume, pitch, pause, num_step, cfg):
    parts = []
    if voice and voice != "Tự động":
        voice_map = {"Mai (Nữ) - Giọng chuẩn": "female, young adult, moderate pitch",
            "Nam (Nam) - Giọng chuẩn": "male, young adult, moderate pitch",
            "Linh (Nữ) - Giọng ấm": "female, middle-aged, warm tone",
            "Minh (Nam) - Giọng trầm": "male, middle-aged, low pitch",
            "Hà (Nữ) - Giọng trẻ": "female, teenager, high pitch",
            "Tùng (Nam) - Giọng đọc": "male, young adult, narration style"}
        if voice in voice_map: parts.append(voice_map[voice])
    if style and style != "Tự nhiên":
        style_map = {"Trang trọng": "formal, professional", "Trò chuyện": "casual, conversational",
            "Đọc truyện": "storytelling, expressive", "Thuyết minh": "narration, documentary style",
            "Quảng cáo": "promotional, energetic"}
        if style in style_map: parts.append(style_map[style])
    instruct = ", ".join(parts) if parts else None
    language = _DISPLAY_TO_CODE.get(lang, None)
    if language == "Auto": language = None
    result, msg = _gen_core(text, language, None, instruct, num_step, cfg, True, speed, None, True, True)
    if result:
        sr, wav = result
        vol = float(volume) if volume else 0.8
        wav = (wav.astype(np.float32) / 32767 * vol * 32767).astype(np.int16)
        return (sr, wav), msg
    return None, msg

def _gen_clone(text, lang, ref_audio, ref_text, speed, volume, num_step, cfg):
    language = _DISPLAY_TO_CODE.get(lang, None)
    if language == "Auto": language = None
    result, msg = _gen_core(text, language, ref_audio, None, num_step, cfg, True, speed, None, True, True, ref_text)
    if result:
        sr, wav = result
        vol = float(volume) if volume else 0.8
        wav = (wav.astype(np.float32) / 32767 * vol * 32767).astype(np.int16)
        return (sr, wav), msg
    return None, msg

def _gen_design(text, lang, preset, accent, custom, speed, volume, num_step, cfg):
    parts = []
    if preset and preset != "Không chọn" and preset in _DESIGN_PRESETS:
        v = _DESIGN_PRESETS[preset]
        if v: parts.append(v)
    if accent and accent != "Không chọn" and accent in _ACCENT_PRESETS:
        v = _ACCENT_PRESETS[accent]
        if v: parts.append(v)
    if custom and custom.strip(): parts.append(custom.strip())
    instruct = ", ".join(parts) if parts else None
    language = _DISPLAY_TO_CODE.get(lang, None)
    if language == "Auto": language = None
    result, msg = _gen_core(text, language, None, instruct, num_step, cfg, True, speed, None, True, True)
    if result:
        sr, wav = result
        vol = float(volume) if volume else 0.8
        wav = (wav.astype(np.float32) / 32767 * vol * 32767).astype(np.int16)
        return (sr, wav), msg
    return None, msg

CUSTOM_CSS = """
* {box-sizing: border-box !important;}
body {background: #f8f9fc !important; font-family: 'Segoe UI', system-ui, sans-serif !important;}
.main-wrap {max-width: 900px !important; margin: 0 auto !important;}
.gradio-container {background: transparent !important; box-shadow: none !important; border-radius: 0 !important; max-width: 100% !important;}
.hero-header {background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%); border-radius: 20px; padding: 35px 40px 30px; position: relative; overflow: hidden;}
.hero-header::after {content: ''; position: absolute; right: 40px; top: 50%; transform: translateY(-50%); width: 180px; height: 180px; background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%); border-radius: 50%;}
.hero-title {font-size: 2em !important; font-weight: 800 !important; color: white !important; margin: 0 0 6px 0 !important;}
.hero-sub {color: rgba(255,255,255,0.85) !important; font-size: 0.95em !important; margin: 0 0 24px 0 !important;}
.features-row {display: flex !important; gap: 16px !important; flex-wrap: wrap !important;}
.feature-item {flex: 1 !important; min-width: 120px !important; text-align: center !important; background: rgba(255,255,255,0.12) !important; border-radius: 12px !important; padding: 14px 8px !important; backdrop-filter: blur(10px) !important;}
.feature-icon {width: 40px !important; height: 40px !important; border-radius: 10px !important; display: flex !important; align-items: center !important; justify-content: center !important; margin: 0 auto 8px !important; font-size: 1.2em !important;}
.feature-icon.yellow {background: #fef3c7 !important;} .feature-icon.blue {background: #dbeafe !important;}
.feature-icon.green {background: #d1fae5 !important;} .feature-icon.purple {background: #ede9fe !important;}
.feature-label {font-size: 0.85em !important; font-weight: 700 !important; color: white !important; margin: 0 0 2px 0 !important;}
.feature-desc {font-size: 0.72em !important; color: rgba(255,255,255,0.7) !important; margin: 0 !important;}
.steps-bar {display: flex !important; align-items: center !important; justify-content: center !important; background: white !important; border-radius: 16px !important; padding: 20px 30px !important; margin: 20px 0 !important; box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;}
.step-item {display: flex !important; align-items: center !important; gap: 10px !important;}
.step-circle {width: 36px !important; height: 36px !important; border-radius: 50% !important; background: #6366f1 !important; color: white !important; display: flex !important; align-items: center !important; justify-content: center !important; font-weight: 700 !important; font-size: 0.9em !important; flex-shrink: 0 !important;}
.step-circle.inactive {background: #e5e7eb !important; color: #9ca3af !important;}
.step-info {text-align: left !important;} .step-title {font-size: 0.9em !important; font-weight: 700 !important; color: #1f2937 !important; margin: 0 !important;}
.step-desc {font-size: 0.75em !important; color: #9ca3af !important; margin: 0 !important;}
.step-dots {flex: 1 !important; border-top: 2px dashed #d1d5db !important; margin: 0 16px !important; min-width: 40px !important;}
.card {background: white !important; border-radius: 16px !important; padding: 28px 30px !important; margin-bottom: 16px !important; box-shadow: 0 2px 12px rgba(0,0,0,0.05) !important; border: 1px solid #f0f0f5 !important;}
.card-header {display: flex !important; align-items: center !important; gap: 10px !important; margin-bottom: 16px !important;}
.card-icon {width: 32px !important; height: 32px !important; border-radius: 8px !important; background: #ede9fe !important; display: flex !important; align-items: center !important; justify-content: center !important; font-size: 1em !important; flex-shrink: 0 !important;}
.card-icon.green-bg {background: #d1fae5 !important;} .card-icon.blue-bg {background: #dbeafe !important;} .card-icon.orange-bg {background: #fed7aa !important;}
.card-title {font-size: 1.1em !important; font-weight: 700 !important; color: #1f2937 !important; margin: 0 !important;}
.card-subtitle {font-size: 0.8em !important; color: #9ca3af !important; margin: 2px 0 0 0 !important;}
.tabs > .tab-nav {background: white !important; border-radius: 14px !important; padding: 6px !important; gap: 6px !important; box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important; border: none !important; margin-bottom: 20px !important;}
.tabs > .tab-nav > button {border-radius: 10px !important; font-weight: 600 !important; font-size: 0.9em !important; padding: 12px 20px !important; border: none !important; color: #6b7280 !important; transition: all 0.2s !important;}
.tabs > .tab-nav > button.selected {background: #6366f1 !important; color: white !important; box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important;}
.tabs > .tab-nav > button:hover:not(.selected) {background: #f3f4f6 !important;}
.gen-btn {width: 100% !important; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important; color: white !important; border: none !important; border-radius: 16px !important; padding: 18px 30px !important; font-size: 1.15em !important; font-weight: 700 !important; cursor: pointer !important; box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important; transition: all 0.3s !important;}
.gen-btn:hover {transform: translateY(-2px) !important; box-shadow: 0 6px 25px rgba(99,102,241,0.5) !important;}
.privacy-note {text-align: center !important; font-size: 0.78em !important; color: #9ca3af !important; margin: 8px 0 20px 0 !important;}
.download-row {display: flex !important; justify-content: center !important; gap: 16px !important; margin-top: 16px !important; flex-wrap: wrap !important;}
.dl-btn {display: flex !important; align-items: center !important; gap: 6px !important; font-size: 0.85em !important; color: #6366f1 !important; cursor: pointer !important; font-weight: 600 !important;}
.tip-note {text-align: center !important; font-size: 0.78em !important; color: #9ca3af !important; margin-top: 20px !important; padding: 10px !important;}
.api-key-box {background: #f0fdf4 !important; border: 2px solid #86efac !important; border-radius: 12px !important; padding: 14px !important; font-family: monospace !important; font-size: 1.05em !important; word-break: break-all !important; color: #166534 !important;}
section {border: none !important; padding: 0 !important; margin: 0 !important;}
"""

with gr.Blocks(css=CUSTOM_CSS, title="Chuyển Văn Bản Thành Giọng Nói") as demo:

    # HEADER
    gr.HTML("""<div class="hero-header">
        <div class="hero-title">Chuyển Văn Bản Thành Giọng Nói</div>
        <div class="hero-sub">AI chuyển văn bản thành giọng nói tự nhiên, chân thực như người thật</div>
        <div class="features-row">
            <div class="feature-item"><div class="feature-icon yellow">⚡</div><div class="feature-label">Giọng nói tự nhiên</div><div class="feature-desc">Như người thật</div></div>
            <div class="feature-item"><div class="feature-icon blue">🧠</div><div class="feature-label">AI thông minh</div><div class="feature-desc">Xử lý nhanh chóng</div></div>
            <div class="feature-item"><div class="feature-icon green">🌐</div><div class="feature-label">Đa ngôn ngữ</div><div class="feature-desc">Hỗ trợ 600+ ngôn ngữ</div></div>
            <div class="feature-item"><div class="feature-icon purple">🛡️</div><div class="feature-label">Bảo mật tuyệt đối</div><div class="feature-desc">Miễn phí, mã nguồn mở</div></div>
        </div>
    </div>""")

    # STEPS
    gr.HTML("""<div class="steps-bar">
        <div class="step-item"><div class="step-circle">1</div><div class="step-info"><div class="step-title">Nhập văn bản</div><div class="step-desc">Hoặc dán nội dung</div></div></div>
        <div class="step-dots"></div>
        <div class="step-item"><div class="step-circle inactive">2</div><div class="step-info"><div class="step-title">Chọn giọng nói</div><div class="step-desc">Thiết lập giọng đọc</div></div></div>
        <div class="step-dots"></div>
        <div class="step-item"><div class="step-circle inactive">3</div><div class="step-info"><div class="step-title">Tạo giọng nói</div><div class="step-desc">Tải về hoặc chia sẻ</div></div></div>
    </div>""")

    # TABS
    with gr.Tabs():
        # ===================== TAB 1: TTS =====================
        with gr.TabItem("⚡ TTS đa ngôn ngữ", id="tts"):
            gr.HTML("""<div class="card"><div class="card-header"><div class="card-icon">📝</div><div>
                <div class="card-title">Nhập văn bản của bạn</div>
                <div class="card-subtitle">Nhập hoặc dán nội dung bạn muốn chuyển thành giọng nói</div></div></div>""")
            tts_text = gr.Textbox(label="", lines=5, show_label=False,
                placeholder="Nhập văn bản tại đây...\n\nGợi ý:\n• Xin chào, hôm nay thời tiết thật đẹp!\n• Trí tuệ nhân tạo đang thay đổi thế giới.\n• Thành công là kết quả của sự kiên trì mỗi ngày.")
            gr.HTML("</div>")

            gr.HTML("""<div class="card"><div class="card-header"><div class="card-icon">🗣️</div><div>
                <div class="card-title">Ngôn ngữ & Giọng nói</div>
                <div class="card-subtitle">Chọn ngôn ngữ và giọng đọc phù hợp</div></div></div>""")
            with gr.Row():
                tts_lang = gr.Dropdown(label="🌐 Ngôn ngữ", choices=_VI_LANGUAGES, value="vi")
                tts_voice = gr.Dropdown(label="🎙️ Giọng nói", choices=_VOICE_OPTIONS, value="Tự động")
            with gr.Row():
                tts_style = gr.Dropdown(label="🎨 Phong cách", choices=_STYLE_OPTIONS, value="Tự nhiên")
            gr.HTML("</div>")

            gr.HTML("""<div class="card"><div class="card-header"><div class="card-icon">⚙️</div><div>
                <div class="card-title">Tùy chỉnh</div></div></div>""")
            with gr.Row():
                tts_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Tốc độ đọc")
                tts_vol = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Âm lượng")
            with gr.Row():
                tts_pitch = gr.Slider(-20, 20, value=0, step=1, label="Cao độ (Pitch)")
                tts_pause = gr.Dropdown(label="Ngắt nghỉ", choices=_PAUSE_OPTIONS, value="Tự động")
            with gr.Accordion("⚙️ Nâng cao", open=False):
                with gr.Row():
                    tts_ns = gr.Slider(4, 64, value=32, step=1, label="Bước suy luận")
                    tts_gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="CFG Scale")
            gr.HTML("</div>")

            def _tts_fn(text, lang, voice, style, speed, vol, pitch, pause, ns, gs):
                return _gen_tts(text, lang, voice, style, speed, vol, pitch, pause, ns, gs)

            tts_btn = gr.Button("🔊 TẠO GIỌNG NÓI", elem_classes="gen-btn")
            gr.HTML('<div class="privacy-note">🔒 Dữ liệu của bạn được bảo mật và sẽ không được lưu trữ</div>')

        # ===================== TAB 2: CLONE =====================
        with gr.TabItem("🎤 Clone giọng nói", id="clone"):
            gr.HTML("""<div class="card"><div class="card-header"><div class="card-icon">📝</div><div>
                <div class="card-title">Nhập văn bản</div>
                <div class="card-subtitle">Nhập nội dung bạn muốn đọc bằng giọng đã clone</div></div></div>""")
            clone_text = gr.Textbox(label="", lines=4, show_label=False,
                placeholder="Nhập nội dung bạn muốn chuyển thành giọng nói...")
            gr.HTML("</div>")

            gr.HTML("""<div class="card"><div class="card-header"><div class="card-icon green-bg">🎧</div><div>
                <div class="card-title">Audio tham chiếu</div>
                <div class="card-subtitle">Tải lên file audio 3-10 giây để clone giọng</div></div></div>""")
            clone_audio = gr.Audio(label="Tải lên audio (tùy chọn)", type="filepath")
            gr.HTML('<div style="font-size:0.82em;color:#6b7280;margin:-5px 0 10px 0;">💡 3-10 giây là tối ưu. Audio >20s sẽ bị cắt tự động. Không có audio = mô hình tự chọn giọng.</div>')
            clone_ref_text = gr.Textbox(label="Nội dung trong audio (tùy chọn)", lines=2,
                placeholder="Để trống sẽ tự động nhận dạng bằng AI")
            clone_lang = gr.Dropdown(label="🌐 Ngôn ngữ đầu ra", choices=_VI_LANGUAGES, value="vi")
            gr.HTML("</div>")

            gr.HTML("""<div class="card"><div class="card-header"><div class="card-icon">⚙️</div><div>
                <div class="card-title">Tùy chỉnh</div></div></div>""")
            with gr.Row():
                clone_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Tốc độ")
                clone_vol = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Âm lượng")
            with gr.Row():
                clone_ns = gr.Slider(4, 64, value=32, step=1, label="Bước suy luận")
                clone_gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="CFG")
            gr.HTML("</div>")

            def _clone_fn(text, lang, ref_audio, ref_text, speed, vol, ns, gs):
                return _gen_clone(text, lang, ref_audio, ref_text, speed, vol, ns, gs)

            clone_btn = gr.Button("🔊 TẠO GIỌNG NÓI", elem_classes="gen-btn")
            gr.HTML('<div class="privacy-note">🔒 Dữ liệu của bạn được bảo mật và sẽ không được lưu trữ</div>')

        # ===================== TAB 3: DESIGN =====================
        with gr.TabItem("🎨 Thiết kế giọng", id="design"):
            gr.HTML("""<div class="card"><div class="card-header"><div class="card-icon">📝</div><div>
                <div class="card-title">Nhập văn bản</div>
                <div class="card-subtitle">Nhập nội dung bạn muốn chuyển thành giọng nói</div></div></div>""")
            design_text = gr.Textbox(label="", lines=4, show_label=False,
                placeholder="Nhập nội dung bạn muốn chuyển thành giọng nói...")
            gr.HTML("</div>")

            gr.HTML("""<div class="card"><div class="card-header"><div class="card-icon orange-bg">🎨</div><div>
                <div class="card-title">Thiết kế giọng nói</div>
                <div class="card-subtitle">Tùy chỉnh loại giọng, vùng miền và phong cách</div></div></div>""")
            with gr.Row():
                design_preset = gr.Dropdown(label="🎭 Loại giọng", choices=["Không chọn"] + list(_DESIGN_PRESETS.keys()), value="Không chọn")
                design_accent = gr.Dropdown(label="🌍 Giọng vùng miền", choices=list(_ACCENT_PRESETS.keys()), value="Không chọn")
            design_custom = gr.Textbox(label="Hoặc tự nhập mô tả (tiếng ANH)", lines=2,
                placeholder="VD: female, young, high pitch, british accent")
            design_lang = gr.Dropdown(label="🌐 Ngôn ngữ đầu ra", choices=_VI_LANGUAGES, value="vi")
            gr.HTML("</div>")

            gr.HTML("""<div class="card"><div class="card-header"><div class="card-icon">⚙️</div><div>
                <div class="card-title">Tùy chỉnh</div></div></div>""")
            with gr.Row():
                design_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Tốc độ")
                design_vol = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Âm lượng")
            with gr.Row():
                design_ns = gr.Slider(4, 64, value=32, step=1, label="Bước suy luận")
                design_gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="CFG")
            gr.HTML("</div>")

            def _design_fn(text, lang, preset, accent, custom, speed, vol, ns, gs):
                return _gen_design(text, lang, preset, accent, custom, speed, vol, ns, gs)

            design_btn = gr.Button("🔊 TẠO GIỌNG NÓI", elem_classes="gen-btn")
            gr.HTML('<div class="privacy-note">🔒 Dữ liệu của bạn được bảo mật và sẽ không được lưu trữ</div>')

        # ===================== TAB 4: API =====================
        with gr.TabItem("🔑 API", id="api"):
            gr.HTML("""<div class="card"><div class="card-header"><div class="card-icon blue-bg">🔑</div><div>
                <div class="card-title">Tạo mã API</div>
                <div class="card-subtitle">Nhập thông tin, nhận mã, dùng ngay</div></div></div>""")
            with gr.Row():
                api_name = gr.Textbox(label="Tên người dùng", placeholder="VD: Nguyễn Văn A")
                api_purpose = gr.Dropdown(label="Mục đích sử dụng", choices=list(_PURPOSE_MAP.keys()), value="Sử dụng cá nhân")
            api_btn = gr.Button("🔑 TẠO MÃ", variant="primary")
            api_key_out = gr.Textbox(label="API Key của bạn", interactive=False, elem_classes="api-key-box")
            api_status = gr.Textbox(label="", interactive=False, show_label=False)
            gr.HTML("</div>")

            gr.HTML("""<div class="card"><div class="card-header"><div class="card-icon">📖</div><div>
                <div class="card-title">Cách dùng API</div></div></div>
                <p style="font-size:0.9em;color:#374151;margin-bottom:10px;"><b>Bước 1:</b> Nhập tên + chọn mục đích → nhấn "Tạo mã"</p>
                <p style="font-size:0.9em;color:#374151;margin-bottom:10px;"><b>Bước 2:</b> Gửi mã cho bên cần dùng</p>
                <p style="font-size:0.9em;color:#374151;margin-bottom:15px;"><b>Bước 3:</b> Bên nhận dùng code bên dưới:</p>
                <p style="font-size:0.85em;color:#6b7280;margin-bottom:5px;"><b>🐍 Python:</b></p>
                <pre style="background:#f3f4f6;padding:12px;border-radius:8px;font-size:0.82em;overflow-x:auto;">from gradio_client import Client
client = Client("Daodat242/omnivoice-tieng-viet")
result = client.predict(
    text="Xin chào!",
    lang="Tiếng Việt",
    api_name="/_vietnamese_fn"
)
print(result[1])</pre>
                <p style="font-size:0.85em;color:#6b7280;margin-bottom:5px;margin-top:12px;"><b>⬛ cURL:</b></p>
                <pre style="background:#f3f4f6;padding:12px;border-radius:8px;font-size:0.82em;overflow-x:auto;">curl -X POST "https://daodat242-omnivoice-tieng-viet.hf.space/_vietnamese_fn" \\
  -H "Content-Type: application/json" \\
  -d '{"data": ["Xin chào!", "Tiếng Việt"]}'</pre>
                <p style="font-size:0.85em;color:#6b7280;margin-bottom:5px;margin-top:12px;"><b>🟨 JavaScript:</b></p>
                <pre style="background:#f3f4f6;padding:12px;border-radius:8px;font-size:0.82em;overflow-x:auto;">const res = await fetch(
  "https://daodat242-omnivoice-tieng-viet.hf.space/_vietnamese_fn",
  { method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({data: ["Xin chào!", "Tiếng Việt"]}) }
);
const result = await res.json();</pre>
            </div>""")

    # ===================== RESULT (shared) =====================
    gr.HTML("""<div class="card"><div class="card-header"><div class="card-icon">🔊</div><div>
        <div class="card-title">Kết quả</div>
        <div class="card-subtitle">Nghe và tải về file giọng nói</div></div></div>""")
    output_audio = gr.Audio(label="", type="numpy")
    output_status = gr.Textbox(label="", lines=1, interactive=False, show_label=False, placeholder="Trạng thái...")
    gr.HTML("""<div class="download-row"><span class="dl-btn">📥 Tải xuống WAV</span></div>""")
    gr.HTML("""<div class="tip-note">💡 Mẹo: Sử dụng dấu câu để tạo ngắt nghỉ tự nhiên. Dấu phẩy (,) ngắt ngắn, dấu chấm (.) ngắt dài.</div>""")

    # ===================== HANDLERS =====================
    tts_btn.click(_tts_fn,
        inputs=[tts_text, tts_lang, tts_voice, tts_style, tts_speed, tts_vol, tts_pitch, tts_pause, tts_ns, tts_gs],
        outputs=[output_audio, output_status], api_name="tts")

    clone_btn.click(_clone_fn,
        inputs=[clone_text, clone_lang, clone_audio, clone_ref_text, clone_speed, clone_vol, clone_ns, clone_gs],
        outputs=[output_audio, output_status], api_name="clone")

    design_btn.click(_design_fn,
        inputs=[design_text, design_lang, design_preset, design_accent, design_custom, design_speed, design_vol, design_ns, design_gs],
        outputs=[output_audio, output_status], api_name="design")

    def _gen_key(name, purpose):
        if not name or not name.strip(): return "", "⚠️ Vui lòng nhập tên"
        return generate_api_key(name, purpose), "✅ Mã đã tạo!"
    api_btn.click(_gen_key, inputs=[api_name, api_purpose], outputs=[api_key_out, api_status])

if __name__ == "__main__":
    demo.queue().launch()
