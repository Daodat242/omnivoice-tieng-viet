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

_VI_LANGUAGES = ["Auto", "vi", "en", "zh", "ja", "ko", "fr", "de", "es", "th", "id", "ms"] + \
    [l for l in _ALL_LANGUAGES if l not in ["Auto", "vi", "en", "zh", "ja", "ko", "fr", "de", "es", "th", "id", "ms"]]

# Map ten day du -> ma ngon ngu
_LANG_DISPLAY = {
    "Auto": "Auto (Tự nhận diện)",
    "vi": "Tiếng Việt",
    "en": "English (Tiếng Anh)",
    "zh": "中文 (Tiếng Trung)",
    "ja": "日本語 (Tiếng Nhật)",
    "ko": "한국어 (Tiếng Hàn)",
    "fr": "Français (Tiếng Pháp)",
    "de": "Deutsch (Tiếng Đức)",
    "es": "Español (Tiếng Tây Ban Nha)",
    "th": "ไทย (Tiếng Thái)",
    "id": "Bahasa Indonesia (Tiếng Indonesia)",
    "ms": "Bahasa Melayu (Tiếng Mã Lai)",
    "pt": "Português (Tiếng Bồ Đào Nha)",
    "it": "Italiano (Tiếng Ý)",
    "ru": "Русский (Tiếng Nga)",
    "ar": "العربية (Tiếng Ả Rập)",
    "hi": "हिन्दी (Tiếng Hindi)",
    "tr": "Türkçe (Tiếng Thổ Nhĩ Kỳ)",
    "nl": "Nederlands (Tiếng Hà Lan)",
    "pl": "Polski (Tiếng Ba Lan)",
    "sv": "Svenska (Tiếng Thụy Điển)",
    "da": "Dansk (Tiếng Đan Mạch)",
    "fi": "Suomi (Tiếng Phần Lan)",
    "nb": "Norsk bokmål (Tiếng Na Uy)",
    "uk": "Українська (Tiếng Ukraina)",
    "cs": "Čeština (Tiếng Séc)",
    "el": "Ελληνικά (Tiếng Hy Lạp)",
    "hu": "Magyar (Tiếng Hungary)",
    "ro": "Română (Tiếng Romania)",
    "bg": "Български (Tiếng Bulgaria)",
    "hr": "Hrvatski (Tiếng Croatia)",
    "sk": "Slovenčina (Tiếng Slovak)",
    "lt": "Lietuvių (Tiếng Litva)",
    "lv": "Latviešu (Tiếng Latvia)",
    "et": "Eesti (Tiếng Estonia)",
    "sl": "Slovenščina (Tiếng Slovenia)",
    "ca": "Català (Tiếng Catalonia)",
    "gl": "Galego (Tiếng Galicia)",
    "eu": "Euskara (Tiếng Basque)",
    "af": "Afrikaans (Tiếng Afrikaans)",
    "sw": "Kiswahili (Tiếng Swahili)",
    "bn": "বাংলা (Tiếng Bengal)",
    "ta": "தமிழ் (Tiếng Tamil)",
    "te": "తెలుగు (Tiếng Telugu)",
    "ml": "മലയാളം (Tiếng Malayalam)",
    "kn": "ಕನ್ನಡ (Tiếng Kannada)",
    "mr": "मराठी (Tiếng Marathi)",
    "gu": "ગુજરાતી (Tiếng Gujarati)",
    "pa": "ਪੰਜਾਬੀ (Tiếng Punjab)",
    "ur": "اردو (Tiếng Urdu)",
    "fa": "فارسی (Tiếng Ba Tư)",
    "he": "עברית (Tiếng Do Thái)",
    "my": "မြန်မာ (Tiếng Myanmar)",
    "km": "ខ្មែរ (Tiếng Khmer)",
    "lo": "ລາວ (Tiếng Lào)",
    "ka": "ქართული (Tiếng Gruzia)",
    "am": "አማርኛ (Tiếng Amharic)",
    "ne": "नेपाली (Tiếng Nepal)",
    "si": "සිංහල (Tiếng Sinhala)",
    "jv": "Basa Jawa (Tiếng Java)",
    "su": "Basa Sunda (Tiếng Sunda)",
    "mg": "Malagasy (Tiếng Malagasy)",
    "mt": "Malti (Tiếng Malta)",
    "cy": "Cymraeg (Tiếng Wales)",
    "ga": "Gaeilge (Tiếng Ireland)",
    "is": "Íslenska (Tiếng Iceland)",
    "mk": "Македонски (Tiếng Macedonia)",
    "sq": "Shqip (Tiếng Albania)",
    "bs": "Bosanski (Tiếng Bosnia)",
    "sr": "Српски (Tiếng Serbia)",
    "me": "Crnogorski (Tiếng Montenegro)",
    "mn": "Монгол (Tiếng Mông Cổ)",
    "ky": "Кыргызча (Tiếng Kyrgyz)",
    "uz": "O'zbek (Tiếng Uzbek)",
    "tk": "Türkmen (Tiếng Turkmen)",
    "ha": "Hausa (Tiếng Hausa)",
    "yo": "Yorùbá (Tiếng Yoruba)",
    "ig": "Igbo (Tiếng Igbo)",
    "zu": "isiZulu (Tiếng Zulu)",
    "xh": "isiXhosa (Tiếng Xhosa)",
    "af": "Afrikaans (Tiếng Afrikaans)",
    "tl": "Filipino (Tiếng Philippines)",
}

# Tao danh sach hien thi voi ten day du
_VI_LANGUAGES_DISPLAY = []
for l in _VI_LANGUAGES:
    _VI_LANGUAGES_DISPLAY.append(_LANG_DISPLAY.get(l, l))

# Map display name -> code
_DISPLAY_TO_CODE = {}
for l in _VI_LANGUAGES:
    display = _LANG_DISPLAY.get(l, l)
    _DISPLAY_TO_CODE[display] = l

_VOICE_PRESETS = {
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

_ACCENT_PRESETS = {
    "Không chọn": "",
    "Giọng Mỹ": "american accent",
    "Giọng Anh": "british accent",
    "Giọng Úc": "australian accent",
    "Giọng Ấn Độ": "indian accent",
    "Giọng Nhật Bản": "japanese accent",
    "Giọng Hàn Quốc": "korean accent",
    "Giọng Nga": "russian accent",
    "Giọng Trung Quốc": "chinese accent",
}

_DIALECT_PRESETS = {
    "Không chọn": "",
    "Hà Nam": "河南话", "Thiểm Tây": "陕西话", "Tứ Xuyên": "四川话",
    "Đông Bắc": "东北话", "Vân Nam": "云南话", "Quảng Đông": "桂林话",
}

_VI_PITCH_MAP = {
    "Cực thấp": "very low pitch", "Thấp": "low pitch",
    "Trung bình": "moderate pitch", "Cao": "high pitch", "Cực cao": "very high pitch",
}

def _gen_core(text, language, ref_audio, instruct, num_step, guidance_scale,
              denoise, speed, duration, preprocess_prompt, postprocess_output,
              mode, ref_text=None):
    if not text or not text.strip():
        return None, "⚠️ Vui lòng nhập văn bản"

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )
    lang = language if (language and language != "Auto") else None
    kw: Dict[str, Any] = dict(text=text.strip(), language=lang, generation_config=gen_config)
    if speed is not None and float(speed) != 1.0: kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0: kw["duration"] = float(duration)
    if mode == "clone" and ref_audio:
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    if instruct and instruct.strip(): kw["instruct"] = instruct.strip()
    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"❌ Lỗi: {type(e).__name__}: {e}"
    waveform = (audio[0] * 32767).astype(np.int16)
    return (sampling_rate, waveform), "✅ Hoàn tất!"

# ============================================================
# UI
# ============================================================
CUSTOM_CSS = """
/* === MAIN === */
.gradio-container {max-width: 1000px !important; margin: auto !important; padding-top: 10px !important;}
main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; min-height: 100vh;}
.gradio-container {background: rgba(255,255,255,0.97) !important; border-radius: 20px !important; box-shadow: 0 20px 60px rgba(0,0,0,0.3) !important;}

/* === HEADER === */
.header-title {text-align: center; font-size: 2.2em !important; font-weight: 800 !important;
    background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; margin-bottom: 0 !important; padding-top: 20px;}
.header-sub {text-align: center; color: #6b7280; font-size: 1em !important; margin-top: 5px !important;}
.header-badge {text-align: center; margin-top: 8px !important;}
.header-badge span {display: inline-block; background: #f0f0ff; color: #667eea; padding: 4px 14px;
    border-radius: 20px; font-size: 0.85em !important; margin: 0 4px; font-weight: 500;}

/* === TABS === */
.tabs > .tab-nav {border-bottom: 2px solid #e5e7eb !important; gap: 0 !important;}
.tabs > .tab-nav > button {font-size: 1em !important; font-weight: 600 !important; padding: 12px 24px !important;
    border-bottom: 3px solid transparent !important; border-radius: 10px 10px 0 0 !important;
    transition: all 0.2s !important; color: #6b7280 !important;}
.tabs > .tab-nav > button.selected {color: #667eea !important; border-bottom-color: #667eea !important;
    background: rgba(102,126,234,0.05) !important;}
.tabs > .tab-nav > button:hover:not(.selected) {color: #374151 !important; background: rgba(0,0,0,0.02) !important;}

/* === BUTTONS === */
.primary-btn {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important; font-size: 1.1em !important; font-weight: 700 !important;
    padding: 14px 40px !important; border: none !important; border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important; transition: all 0.3s !important;
    letter-spacing: 0.5px !important;}
.primary-btn:hover {transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(102,126,234,0.5) !important;}
.primary-btn:active {transform: translateY(0) !important;}

/* === INPUTS === */
.input-card {background: #f9fafb !important; border: 1px solid #e5e7eb !important; border-radius: 16px !important;
    padding: 20px !important; margin: 8px 0 !important;}
.output-card {background: #f0f9ff !important; border: 2px solid #bfdbfe !important; border-radius: 16px !important;
    padding: 20px !important; margin: 8px 0 !important;}
section {border: none !important;}

/* === STEP INDICATOR === */
.step-box {display: flex; align-items: center; gap: 10px; margin: 10px 0 15px 0; padding: 10px 16px;
    background: #f8f9ff; border-radius: 10px; border-left: 4px solid #667eea;}
.step-num {display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px;
    background: #667eea; color: white; border-radius: 50%; font-weight: 700; font-size: 0.85em; flex-shrink: 0;}
.step-text {color: #374151; font-size: 0.95em;}

/* === STATUS === */
.status-ok {background: #ecfdf5 !important; border: 1px solid #a7f3d0 !important; border-radius: 10px !important;}
.status-err {background: #fef2f2 !important; border: 1px solid #fecaca !important; border-radius: 10px !important;}

/* === AUDIO PLAYER === */
audio {border-radius: 10px !important;}

/* === SECTION TITLE === */
.section-title {font-size: 1.15em !important; font-weight: 700 !important; color: #374151 !important;
    margin: 15px 0 10px 0 !important; padding-bottom: 6px !important; border-bottom: 2px solid #e5e7eb;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=CUSTOM_CSS,
               title="Chuyển Văn Bản Thành Giọng Nói") as demo:

    # ===================== HEADER =====================
    gr.HTML("""
    <div style="text-align:center; padding: 10px 0 0 0;">
        <div class="header-title">🎙️ Chuyển Văn Bản Thành Giọng Nói</div>
        <div class="header-sub">Mô hình AI chuyển văn bản thành giọng nói chất lượng cao</div>
        <div class="header-badge">
            <span>🌐 600+ ngôn ngữ</span>
            <span>🎤 Clone giọng</span>
            <span>🎨 Thiết kế giọng</span>
            <span>⚡ Tốc độ cao</span>
        </div>
    </div>
    """)

    with gr.Tabs():
        # ==========================================================
        # TAB 1: TTS - CHUYỂN VĂN BẢN ĐA NGÔN NGỮ
        # ==========================================================
        with gr.TabItem("⚡ TTS", id="tts"):
            gr.HTML('<div class="step-box"><span class="step-num">1</span><span class="step-text">Nhập văn bản</span><span class="step-num">2</span><span class="step-text">Chọn ngôn ngữ & giọng</span><span class="step-num">3</span><span class="step-text">Nhấn Tạo giọng</span></div>')

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">📝 Văn bản (đa ngôn ngữ)</div>')
                    vi_text = gr.Textbox(label="", lines=5, show_label=False,
                        placeholder="Nhập văn bản và CHỌN NGÔN NGỮ bên dưới...\n\nVí dụ:\n  • Tiếng Việt: Xin chào các bạn\n  • Tiếng Anh: Hello, how are you?\n  • Tiếng Trung: 你好，你好吗？\n  • Tiếng Nhật: こんにちは\n  • Tiếng Hàn: 안녕하세요",
                        elem_classes="input-card")

                    vi_lang = gr.Dropdown(label="Ngôn ngữ đầu ra", choices=_VI_LANGUAGES_DISPLAY,
                        value="Auto (Tự nhận diện)", allow_custom_value=False, elem_classes="input-card",
                        info="Chọn đúng ngôn ngữ để phát âm chuẩn nhất!")

                    gr.HTML('<div class="section-title">🎭 Chọn giọng</div>')
                    vi_voice_type = gr.Dropdown(label="Giọng nói", value="Tự động",
                        choices=["Tự động"] + list(_VOICE_PRESETS.keys()),
                        elem_classes="input-card")
                    vi_pitch = gr.Dropdown(label="Âm vực", value="Không chọn",
                        choices=["Không chọn"] + list(_VI_PITCH_MAP.keys()),
                        elem_classes="input-card")

                    gr.HTML('<div class="section-title">🔊 Âm lượng</div>')
                    vi_vol = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Âm lượng",
                        info="1.0 = bình thường, >1 to hơn, <1 nhỏ hơn")

                    with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
                        with gr.Row():
                            vi_sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Tốc độ")
                            vi_du = gr.Number(value=None, label="Thời lượng (giây)")
                        with gr.Row():
                            vi_ns = gr.Slider(4, 64, value=32, step=1, label="Bước suy luận")
                            vi_gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="CFG")

                    vi_btn = gr.Button("▶️  TẠO GIỌNG", variant="primary", elem_classes="primary-btn")

                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">🔊 Kết quả</div>')
                    vi_audio = gr.Audio(label="", type="numpy", elem_classes="output-card")
                    vi_status = gr.Textbox(label="", lines=1, interactive=False, show_label=False,
                        elem_classes="output-card", placeholder="Trạng thái sẽ hiển thị ở đây...")

            def _vietnamese_fn(text, lang, voice_type, pitch, volume, sp, du, ns, gs):
                if not text or not text.strip():
                    return None, "⚠️ Vui lòng nhập văn bản"
                parts = []
                if voice_type and voice_type != "Tự động" and voice_type in _VOICE_PRESETS:
                    parts.append(_VOICE_PRESETS[voice_type])
                if pitch and pitch != "Không chọn" and pitch in _VI_PITCH_MAP:
                    pv = _VI_PITCH_MAP[pitch]
                    if parts:
                        old = parts[0]
                        for p in ["very low pitch","low pitch","moderate pitch","high pitch","very high pitch"]:
                            old = old.replace(p, pv)
                        parts[0] = old
                    else:
                        parts.append("moderate pitch")
                instruct = ", ".join(parts) if parts else None
                # Chuyen display name thanh language code
                language = _DISPLAY_TO_CODE.get(lang, None)
                if language == "Auto": language = None
                gen_config = OmniVoiceGenerationConfig(num_step=int(ns or 32),
                    guidance_scale=float(gs) if gs is not None else 2.0,
                    denoise=True, preprocess_prompt=True, postprocess_output=True)
                kw = dict(text=text.strip(), language=language, generation_config=gen_config)
                if instruct: kw["instruct"] = instruct
                if sp is not None and float(sp) != 1.0: kw["speed"] = float(sp)
                if du is not None and float(du) > 0: kw["duration"] = float(du)
                try:
                    audio = model.generate(**kw)
                except Exception as e:
                    return None, f"❌ Lỗi: {type(e).__name__}: {e}"
                vol = float(volume) if volume else 1.0
                waveform = (audio[0] * 32767 * vol).astype(np.int16)
                return (sampling_rate, waveform), "✅ Hoàn tất! Nhấn ▶ để nghe"

            vi_btn.click(_vietnamese_fn,
                inputs=[vi_text, vi_lang, vi_voice_type, vi_pitch, vi_vol, vi_sp, vi_du, vi_ns, vi_gs],
                outputs=[vi_audio, vi_status], api_name="tts")

        # ==========================================================
        # TAB 2: CLONE GIỌNG NÓI
        # ==========================================================
        with gr.TabItem("🎤 Clone Giọng Nói", id="clone"):
            gr.HTML('<div class="step-box"><span class="step-num">1</span><span class="step-text">Nhập văn bản</span><span class="step-num">2</span><span class="step-text">Tải audio mẫu (tùy chọn)</span><span class="step-num">3</span><span class="step-text">Nhấn Tạo giọng</span></div>')

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">📝 Văn bản</div>')
                    vc_text = gr.Textbox(label="", lines=4, show_label=False,
                        placeholder="Nhập nội dung bạn muốn chuyển thành giọng nói...",
                        elem_classes="input-card")

                    gr.HTML('<div class="section-title">🎧 Audio tham chiếu</div>')
                    vc_ref_audio = gr.Audio(label="Tải lên file audio (3-10 giây, tùy chọn)",
                        type="filepath", elem_classes="input-card")
                    gr.HTML('<div style="font-size:0.85em;color:#6b7280;margin:-5px 0 10px 0;">💡 3-10 giây là tối ưu. Audio >20s sẽ bị cắt tự động. Không có audio = mô hình tự chọn giọng.</div>')
                    vc_ref_text = gr.Textbox(label="Nội dung trong audio (tùy chọn)", lines=2,
                        placeholder="Để trống sẽ tự động nhận dạng bằng AI",
                        elem_classes="input-card")

                    vc_lang = gr.Dropdown(label="Ngôn ngữ đầu ra", choices=_VI_LANGUAGES_DISPLAY,
                        value="Auto (Tự nhận diện)", allow_custom_value=False, elem_classes="input-card")

                    with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
                        with gr.Row():
                            vc_sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Tốc độ")
                            vc_du = gr.Number(value=None, label="Thời lượng (giây)")
                        with gr.Row():
                            vc_ns = gr.Slider(4, 64, value=32, step=1, label="Bước suy luận")
                            vc_gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="CFG")
                        with gr.Row():
                            vc_dn = gr.Checkbox(label="Khử nhiễu", value=True)
                            vc_pp = gr.Checkbox(label="Tiền xử lý", value=True)
                            vc_po = gr.Checkbox(label="Xử lý đầu ra", value=True)
                            vc_instruct = gr.Textbox(label="Chỉ dẫn bổ sung", lines=1)

                    vc_btn = gr.Button("▶️  TẠO GIỌNG", variant="primary", elem_classes="primary-btn")

                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">🔊 Kết quả</div>')
                    vc_audio = gr.Audio(label="", type="numpy", elem_classes="output-card")
                    vc_status = gr.Textbox(label="", lines=1, interactive=False, show_label=False,
                        elem_classes="output-card", placeholder="Trạng thái sẽ hiển thị ở đây...")

            def _clone_fn(text, lang, ref_aud, ref_text, ns, gs, dn, sp, du, pp, po, instruct):
                mode = "clone" if ref_aud else "auto"
                return _gen_core(text, lang, ref_aud, instruct, ns, gs, dn, sp, du, pp, po, mode=mode, ref_text=ref_text or None)

            vc_btn.click(_clone_fn,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po, vc_instruct],
                outputs=[vc_audio, vc_status], api_name="clone")

        # ==========================================================
        # TAB 3: THIẾT KẾ GIỌNG NÓI
        # ==========================================================
        with gr.TabItem("🎨 Thiết Kế Giọng Nói", id="design"):
            gr.HTML('<div class="step-box"><span class="step-num">1</span><span class="step-text">Chọn loại giọng</span><span class="step-num">2</span><span class="step-text">Nhập văn bản</span><span class="step-num">3</span><span class="step-text">Nhấn Tạo giọng</span></div>')

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">📝 Văn bản</div>')
                    vd_text = gr.Textbox(label="", lines=4, show_label=False,
                        placeholder="Nhập nội dung bạn muốn chuyển thành giọng nói...",
                        elem_classes="input-card")

                    gr.HTML('<div class="section-title">🎭 Thiết kế giọng</div>')
                    vd_preset = gr.Dropdown(label="Loại giọng",
                        choices=["Không chọn"] + list(_VOICE_PRESETS.keys()), value="Không chọn",
                        elem_classes="input-card")
                    vd_accent = gr.Dropdown(label="Giọng Anh",
                        choices=list(_ACCENT_PRESETS.keys()), value="Không chọn",
                        elem_classes="input-card")
                    vd_dialect = gr.Dropdown(label="Phương ngữ Trung Quốc",
                        choices=list(_DIALECT_PRESETS.keys()), value="Không chọn",
                        elem_classes="input-card")
                    vd_custom = gr.Textbox(label="Hoặc tự nhập mô tả (tiếng ANH)",
                        lines=2, placeholder="VD: female, young, high pitch",
                        elem_classes="input-card")
                    vd_lang = gr.Dropdown(label="Ngôn ngữ đầu ra", choices=_VI_LANGUAGES_DISPLAY,
                        value="Auto (Tự nhận diện)", allow_custom_value=False, elem_classes="input-card")

                    with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
                        with gr.Row():
                            vd_sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Tốc độ")
                            vd_du = gr.Number(value=None, label="Thời lượng (giây)")
                        with gr.Row():
                            vd_ns = gr.Slider(4, 64, value=32, step=1, label="Bước suy luận")
                            vd_gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="CFG")
                        with gr.Row():
                            vd_dn = gr.Checkbox(label="Khử nhiễu", value=True)
                            vd_pp = gr.Checkbox(label="Tiền xử lý", value=True)
                            vd_po = gr.Checkbox(label="Xử lý đầu ra", value=True)

                    vd_btn = gr.Button("▶️  TẠO GIỌNG", variant="primary", elem_classes="primary-btn")

                with gr.Column(scale=1):
                    gr.HTML('<div class="section-title">🔊 Kết quả</div>')
                    vd_audio = gr.Audio(label="", type="numpy", elem_classes="output-card")
                    vd_status = gr.Textbox(label="", lines=1, interactive=False, show_label=False,
                        elem_classes="output-card", placeholder="Trạng thái sẽ hiển thị ở đây...")

            def _build_design_instruct(preset, accent, dialect, custom):
                parts = []
                if preset and preset != "Không chọn" and preset in _VOICE_PRESETS:
                    v = _VOICE_PRESETS[preset]
                    if v: parts.append(v)
                if accent and accent != "Không chọn" and accent in _ACCENT_PRESETS:
                    v = _ACCENT_PRESETS[accent]
                    if v: parts.append(v)
                if dialect and dialect != "Không chọn" and dialect in _DIALECT_PRESETS:
                    v = _DIALECT_PRESETS[dialect]
                    if v: parts.append(v)
                if custom and custom.strip(): parts.append(custom.strip())
                return ", ".join(parts) if parts else None

            def _design_fn(text, lang, preset, accent, dialect, custom, ns, gs, dn, sp, du, pp, po):
                instruct = _build_design_instruct(preset, accent, dialect, custom)
                return _gen_core(text, lang, None, instruct, ns, gs, dn, sp, du, pp, po, mode="design")

            vd_btn.click(_design_fn,
                inputs=[vd_text, vd_lang, vd_preset, vd_accent, vd_dialect, vd_custom,
                        vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po],
                outputs=[vd_audio, vd_status], api_name="design")

        # ==========================================================
        # TAB 4: HƯỚNG DẪN
        # ==========================================================
        with gr.TabItem("📖 Hướng Dẫn", id="help"):

            gr.Markdown("""
### 🎤 Clone Giọng Nói
1. Nhập văn bản bạn muốn chuyển thành giọng nói
2. **(Tuỳ chọn)** Tải lên file audio tham chiếu 3-10 giây
3. Nhấn **Tạo giọng** và chờ kết quả

> **Mẹo:** Không có audio = mô hình tự chọn giọng phù hợp

---

### 🎨 Thiết Kế Giọng Nói
1. Nhập văn bản cần nói
2. Chọn loại giọng từ danh sách (Nam/Nữ, Trẻ/Già...)
3. Nhấn **Tạo giọng**

> **Lưu ý:** Mô tả giọng phải nhập bằng **tiếng ANH** (VD: `female, young, high pitch`)

---

### ⚙️ Cài Đặt Nâng Cao
| Tham số | Mặc định | Giải thích |
|---------|----------|------------|
| Tốc độ | 1.0 | >1 nhanh hơn, <1 chậm hơn |
| Thời lượng | Trống | Số giây cố định (bỏ qua tốc độ) |
| Bước suy luận | 32 | Thấp = nhanh, Cao = đẹp hơn |
| CFG | 2.0 | Cao = rõ hơn, Thấp = tự nhiên hơn |
| Khử nhiễu | Bật | Xóa tiếng ồn |
| Tiền xử lý | Bật | Xóa im lặng, thêm dấu câu |
| Xử lý đầu ra | Bật | Cắt im lặng dài |

---

*Powered by [OmniVoice](https://github.com/k2-fsa/OmniVoice) - Xiaomi AI Lab*
""")

        # ==========================================================
        # TAB 5: API
        # ==========================================================
        with gr.TabItem("🔌 API", id="api"):
            gr.Markdown("""
### 🌐 API — Dùng OmniVoice từ ứng dụng của bạn

OmniVoice cung cấp API miễn phí để bạn tích hợp chuyển văn bản thành giọng nói vào ứng dụng của mình.

**Base URL:** `https://daodat242-omnivoice-tieng-viet.hf.space`
""")

            with gr.Tabs():
                # --- Python ---
                with gr.TabItem("🐍 Python"):
                    gr.Markdown("""
#### Cách 1: Dùng `gradio_client` (đơn giản nhất)
```bash
pip install gradio_client
```
```python
from gradio_client import Client

client = Client("Daodat242/omnivoice-tieng-viet")

# TTS cơ bản
result = client.predict(
    text="Xin chào các bạn, chúc mọi người một ngày tốt lành!",
    language="Tiếng Việt",
    voice_type="Nam - Trẻ tuổi",
    pitch="Không chọn",
    volume=1.0,
    speed=1.0,
    duration=None,
    num_step=32,
    guidance_scale=2.0,
    api_name="/tts"
)

print(result)  # (sample_rate, numpy_array) hoặc đường dẫn file audio
```

#### Cách 2: Dùng `requests` (REST API)
```python
import requests
import json

url = "https://daodat242-omnivoice-tieng-viet.hf.space/_vietnamese_fn"
payload = {
    "data": [
        "Xin chào!",           # text
        "Tiếng Việt",          # language
        "Tự động",            # voice_type
        "Không chọn",         # pitch
        1.0,                   # volume
        1.0,                   # speed
        None,                  # duration
        32,                    # num_step
        2.0                    # guidance_scale
    ]
}
response = requests.post(url, json=payload)
print(response.json())
```
""")

                # --- cURL ---
                with gr.TabItem("⬛ cURL"):
                    gr.Markdown("""
#### TTS cơ bản
```bash
curl -X POST "https://daodat242-omnivoice-tieng-viet.hf.space/_vietnamese_fn" \\
  -H "Content-Type: application/json" \\
  -d '{
    "data": [
      "Xin chào các bạn!",
      "Tiếng Việt",
      "Nam - Trẻ tuổi",
      "Không chọn",
      1.0,
      1.0,
      null,
      32,
      2.0
    ]
  }'
```

#### Clone giọng nói
```bash
curl -X POST "https://daodat242-omnivoice-tieng-viet.hf.space/_clone_fn" \\
  -H "Content-Type: application/json" \\
  -d '{
    "data": [
      "Xin chào!",
      "Tiếng Việt",
      null,
      null,
      32,
      2.0,
      true,
      1.0,
      null,
      true,
      true,
      null
    ]
  }'
```
""")

                # --- JavaScript ---
                with gr.TabItem("🟨 JavaScript"):
                    gr.Markdown("""
#### Node.js / Browser
```javascript
const response = await fetch(
  "https://daodat242-omnivoice-tieng-viet.hf.space/_vietnamese_fn",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data: [
        "Xin chào các bạn!",    // text
        "Tiếng Việt",           // language
        "Nam - Trẻ tuổi",      // voice_type
        "Không chọn",          // pitch
        1.0,                    // volume
        1.0,                    // speed
        null,                   // duration
        32,                     // num_step
        2.0                     // guidance_scale
      ]
    })
  }
);

const result = await response.json();
console.log(result);
```
""")

                # --- Tham số ---
                with gr.TabItem("📋 Tham số"):
                    gr.Markdown("""
#### Endpoint: `/tts`

| # | Tham số | Kiểu | Mặc định | Mô tả |
|---|---------|------|----------|-------|
| 0 | `text` | string | *bắt buộc* | Văn bản cần chuyển thành giọng nói |
| 1 | `language` | string | `"Auto"` | Ngôn ngữ đầu ra (xem danh sách bên dưới) |
| 2 | `voice_type` | string | `"Tự động"` | Loại giọng: `Nam - Trẻ tuổi`, `Nữ - Trung niên`... |
| 3 | `pitch` | string | `"Không chọn"` | Âm vực: `Thấp`, `Trung bình`, `Cao`... |
| 4 | `volume` | float | `1.0` | Âm lượng (0.1 - 3.0) |
| 5 | `speed` | float | `1.0` | Tốc độ (0.5 - 1.5) |
| 6 | `duration` | float/null | `null` | Thời lượng cố định (giây), bỏ qua speed |
| 7 | `num_step` | int | `32` | Bước suy luận (4-64) |
| 8 | `guidance_scale` | float | `2.0` | CFG (0.0-4.0) |

#### Endpoint: `/clone`

| # | Tham số | Kiểu | Mặc định | Mô tả |
|---|---------|------|----------|-------|
| 0 | `text` | string | *bắt buộc* | Văn bản cần nói |
| 1 | `language` | string | `"Auto"` | Ngôn ngữ đầu ra |
| 2 | `ref_audio` | file/null | `null` | File audio tham chiếu (3-10 giây) |
| 3 | `ref_text` | string/null | `null` | Nội dung trong audio (để trống = tự nhận dạng) |
| 4 | `num_step` | int | `32` | Bước suy luận |
| 5 | `guidance_scale` | float | `2.0` | CFG |
| 6 | `denoise` | bool | `true` | Khử nhiễu |
| 7 | `speed` | float | `1.0` | Tốc độ |
| 8 | `duration` | float/null | `null` | Thời lượng cố định |
| 9 | `preprocess_prompt` | bool | `true` | Tiền xử lý |
| 10 | `postprocess_output` | bool | `true` | Xử lý đầu ra |
| 11 | `instruct` | string/null | `null` | Chỉ dẫn giọng nói (tiếng ANH) |

#### Endpoint: `/design`

| # | Tham số | Kiểu | Mặc định | Mô tả |
|---|---------|------|----------|-------|
| 0 | `text` | string | *bắt buộc* | Văn bản cần nói |
| 1 | `language` | string | `"Auto"` | Ngôn ngữ đầu ra |
| 2 | `preset` | string | `"Không chọn"` | Loại giọng (Nam/Nữ, Trẻ/Già...) |
| 3 | `accent` | string | `"Không chọn"` | Giọng regional (Mỹ, Anh, Úc...) |
| 4 | `dialect` | string | `"Không chọn"` | Phương ngữ Trung Quốc |
| 5 | `custom` | string | `""` | Mô tả tùy chỉnh (tiếng ANH) |
| 6 | `num_step` | int | `32` | Bước suy luận |
| 7 | `guidance_scale` | float | `2.0` | CFG |
| 8 | `denoise` | bool | `true` | Khử nhiễu |
| 9 | `speed` | float | `1.0` | Tốc độ |
| 10 | `duration` | float/null | `null` | Thời lượng cố định |
| 11 | `preprocess_prompt` | bool | `true` | Tiền xử lý |
| 12 | `postprocess_output` | bool | `true` | Xử lý đầu ra |
""")

                # --- Ngôn ngữ ---
                with gr.TabItem("🌐 Ngôn ngữ"):
                    gr.Markdown("""
OmniVoice hỗ trợ **600+ ngôn ngữ**. Dưới đây là các ngôn ngữ phổ biến:

| Mã | Ngôn ngữ | Mã | Ngôn ngữ |
|----|----------|----|----------|
| `vi` | Tiếng Việt | `en` | English |
| `zh` | 中文 (Trung Quốc) | `ja` | 日本語 (Nhật) |
| `ko` | 한국어 (Hàn) | `fr` | Français (Pháp) |
| `de` | Deutsch (Đức) | `es` | Español (Tây Ban Nha) |
| `th` | ไทย (Thái) | `id` | Bahasa Indonesia |
| `pt` | Português (BĐN) | `it` | Italiano (Ý) |
| `ru` | Русский (Nga) | `ar` | Ả Rập |
| `hi` | हिन्दी (Hindi) | `tr` | Türkçe (Thổ Nhĩ Kỳ) |

> **Lưu ý:** Khi dùng API, truyền **tên hiển thị** (VD: `"Tiếng Việt"`) hoặc `null` cho Auto.
""")

            gr.Markdown("""
---
### 📌 Ghi chú quan trọng

- **Miễn phí:** API không mất phí, nhưng bị giới hạn tài nguyên (CPU)
- **Tốc độ:** Mỗi yêu cầu mất 5-30 giây tùy độ dài văn bản
- **Audio:** Trả về file audio (WAV) hoặc numpy array
- **Lỗi:** Trả về `{"error": "..."}` nếu có lỗi
- **Chi tiết:** Xem thêm tại [`docs/api.md`](https://github.com/Daodat242/omnivoice-tieng-viet/blob/main/docs/api.md)
""")

if __name__ == "__main__":
    demo.queue().launch()
