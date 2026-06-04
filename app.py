#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OmniVoice - Chuyển văn bản thành giọng nói (HuggingFace Spaces)"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from typing import Any, Dict

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig

# Tải model
print("Đang tải model OmniVoice...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
print("Model đã tải!")

sampling_rate = model.sampling_rate

# ---------------------------------------------------------------------------
# Ngôn ngữ
# ---------------------------------------------------------------------------
try:
    from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
    _ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)
except Exception:
    _ALL_LANGUAGES = ["Auto", "vi", "en", "zh", "ja", "ko", "fr", "de", "es"]

_VI_POPULAR = ["Auto", "vi", "en", "zh", "ja", "ko", "fr", "de", "es", "th", "id", "ms"]
_VI_LANGUAGES = _VI_POPULAR + [l for l in _ALL_LANGUAGES if l not in _VI_POPULAR]

# ---------------------------------------------------------------------------
# Voice Design presets
# ---------------------------------------------------------------------------
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
    "Phương ngữ Hà Nam": "河南话",
    "Phương ngữ Thiểm Tây": "陕西话",
    "Phương ngữ Tứ Xuyên": "四川话",
    "Phương ngữ Đông Bắc": "东北话",
    "Phương ngữ Vân Nam": "云南话",
    "Phương ngữ Quảng Đông": "桂林话",
}

# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def _gen_core(text, language, ref_audio, instruct, num_step, guidance_scale,
              denoise, speed, duration, preprocess_prompt, postprocess_output,
              mode, ref_text=None):
    if not text or not text.strip():
        return None, "⚠️ Vui lòng nhập văn bản cần chuyển thành giọng nói."

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None
    kw: Dict[str, Any] = dict(text=text.strip(), language=lang, generation_config=gen_config)

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone" and ref_audio:
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=ref_text,
        )

    if instruct and instruct.strip():
        kw["instruct"] = instruct.strip()

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"❌ Lỗi: {type(e).__name__}: {e}"

    waveform = (audio[0] * 32767).astype(np.int16)
    return (sampling_rate, waveform), "✅ Hoàn tất!"

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(primary_hue="blue", font=["Inter", "Arial", "sans-serif"])

css = """
.gradio-container {max-width: 960px !important; margin: auto !important;}
.gradio-container h1 {text-align: center; font-size: 1.8em !important; margin-bottom: 0.2em !important;}
.compact-audio audio {height: 60px !important;}
.generate-btn {font-size: 1.1em !important; padding: 12px 32px !important;}
"""

with gr.Blocks(theme=theme, css=css, title="OmniVoice - Tiếng Việt") as demo:
    gr.Markdown("""
# 🎙️ OmniVoice

Mô hình chuyển văn bản thành giọng nói cho **hơn 600 ngôn ngữ**

*Clone giọng nói từ file audio* | *Thiết kế giọng theo ý muốn* | *Chất lượng cao, tốc độ nhanh*
""")

    with gr.Tabs():
        # ==========================================================
        # TAB 1: Tiếng Việt
        # ==========================================================
        with gr.TabItem("🇻🇳 Tiếng Việt", id="vietnamese"):
            gr.Markdown("> **Bước 1:** Nhập văn bản | **Bước 2:** Chọn giọng | **Bước 3:** Nhấn Tạo giọng")
            with gr.Row():
                with gr.Column(scale=1):
                    vi_text = gr.Textbox(label="📝 Văn bản tiếng Việt", lines=5,
                                         placeholder="Nhập nội dung tiếng Việt bạn muốn chuyển thành giọng nói...")
                    vi_voice_type = gr.Dropdown(label="Giọng nói", value="Tự động",
                        choices=["Tự động"] + list(_VOICE_PRESETS.keys()))
                    vi_pitch = gr.Dropdown(label="Âm vực (tùy chọn)", value="Không chọn",
                        choices=["Không chọn", "Âm vực cực thấp", "Âm vực thấp", "Âm vực trung bình", "Âm vực cao", "Âm vực cực cao"])
                    vi_btn = gr.Button("▶️ Tạo giọng", variant="primary", elem_classes="generate-btn")
                with gr.Column(scale=1):
                    vi_audio = gr.Audio(label="🔊 Kết quả", type="numpy")
                    vi_status = gr.Textbox(label="Trạng thái", lines=1, interactive=False)

            with gr.Accordion("⚙️ Cài đặt (tùy chọn)", open=False):
                with gr.Row():
                    vi_sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Tốc độ")
                    vi_du = gr.Number(value=None, label="Thời lượng (giây)")
                with gr.Row():
                    vi_ns = gr.Slider(4, 64, value=32, step=1, label="Bước suy luận")
                    vi_gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="CFG")

            _VI_PITCH_MAP = {
                "Âm vực cực thấp": "very low pitch", "Âm vực thấp": "low pitch",
                "Âm vực trung bình": "moderate pitch", "Âm vực cao": "high pitch", "Âm vực cực cao": "very high pitch",
            }

            def _vietnamese_fn(text, voice_type, pitch, sp, du, ns, gs):
                if not text or not text.strip():
                    return None, "⚠️ Vui lòng nhập văn bản tiếng Việt."
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
                gen_config = OmniVoiceGenerationConfig(num_step=int(ns or 32),
                    guidance_scale=float(gs) if gs is not None else 2.0,
                    denoise=True, preprocess_prompt=True, postprocess_output=True)
                kw = dict(text=text.strip(), language="vi", generation_config=gen_config)
                if instruct:
                    kw["instruct"] = instruct
                if sp is not None and float(sp) != 1.0:
                    kw["speed"] = float(sp)
                if du is not None and float(du) > 0:
                    kw["duration"] = float(du)
                try:
                    audio = model.generate(**kw)
                except Exception as e:
                    return None, f"❌ Lỗi: {type(e).__name__}: {e}"
                waveform = (audio[0] * 32767).astype(np.int16)
                return (sampling_rate, waveform), "✅ Hoàn tất!"

            vi_btn.click(_vietnamese_fn,
                inputs=[vi_text, vi_voice_type, vi_pitch, vi_sp, vi_du, vi_ns, vi_gs],
                outputs=[vi_audio, vi_status])

        # ==========================================================
        # TAB 2: Clone giọng nói
        # ==========================================================
        with gr.TabItem("🎤 Clone giọng nói", id="clone"):
            gr.Markdown("> **Bước 1:** Nhập văn bản | **Bước 2:** (Tuỳ chọn) Tải audio tham chiếu | **Bước 3:** Nhấn Tạo giọng\n\n*Không có audio tham chiếu = mô hình tự chọn giọng*")
            with gr.Row():
                with gr.Column(scale=1):
                    vc_text = gr.Textbox(label="📝 Văn bản cần nói", lines=3,
                                         placeholder="Nhập nội dung bạn muốn chuyển thành giọng nói...")
                    vc_ref_audio = gr.Audio(label="🎧 Audio tham chiếu (tùy chọn - 3-10 giây)", type="filepath", elem_classes="compact-audio")
                    vc_ref_text = gr.Textbox(label="📄 Văn bản trong audio (tùy chọn)", lines=2,
                                             placeholder="Nhập nội dung trong audio tham chiếu.\nĐể trống sẽ tự động nhận dạng.")
                    vc_lang = gr.Dropdown(label="🌐 Ngôn ngữ đầu ra", choices=_VI_LANGUAGES, value="Auto", allow_custom_value=False)
                    vc_btn = gr.Button("▶️ Tạo giọng", variant="primary", elem_classes="generate-btn")
                with gr.Column(scale=1):
                    vc_audio = gr.Audio(label="🔊 Kết quả", type="numpy")
                    vc_status = gr.Textbox(label="Trạng thái", lines=1, interactive=False)

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

            def _clone_fn(text, lang, ref_aud, ref_text, ns, gs, dn, sp, du, pp, po, instruct):
                mode = "clone" if ref_aud else "auto"
                return _gen_core(text, lang, ref_aud, instruct, ns, gs, dn, sp, du, pp, po, mode=mode, ref_text=ref_text or None)

            vc_btn.click(_clone_fn,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po, vc_instruct],
                outputs=[vc_audio, vc_status])

        # ==========================================================
        # TAB 3: Thiết kế giọng nói
        # ==========================================================
        with gr.TabItem("🎨 Thiết kế giọng nói", id="design"):
            gr.Markdown("> **Bước 1:** Chọn loại giọng | **Bước 2:** Nhập văn bản | **Bước 3:** Nhấn Tạo giọng\n\n*Không cần file audio - mô hình sẽ tạo giọng theo mô tả của bạn*")
            with gr.Row():
                with gr.Column(scale=1):
                    vd_text = gr.Textbox(label="📝 Văn bản cần nói", lines=3,
                                         placeholder="Nhập nội dung bạn muốn chuyển thành giọng nói...")
                    vd_preset = gr.Dropdown(label="Loại giọng", choices=["Không chọn"] + list(_VOICE_PRESETS.keys()), value="Không chọn")
                    vd_accent = gr.Dropdown(label="Giọng Anh", choices=list(_ACCENT_PRESETS.keys()), value="Không chọn")
                    vd_dialect = gr.Dropdown(label="Phương ngữ Trung Quốc", choices=list(_DIALECT_PRESETS.keys()), value="Không chọn")
                    vd_custom = gr.Textbox(label="✏️ Hoặc tự nhập mô tả giọng", lines=2,
                                           placeholder="VD: female, young, high pitch",
                                           info="Nhập bằng tiếng ANH, phân cách bằng dấu phẩy")
                    vd_lang = gr.Dropdown(label="🌐 Ngôn ngữ đầu ra", choices=_VI_LANGUAGES, value="Auto", allow_custom_value=False)
                    vd_btn = gr.Button("▶️ Tạo giọng", variant="primary", elem_classes="generate-btn")
                with gr.Column(scale=1):
                    vd_audio = gr.Audio(label="🔊 Kết quả", type="numpy")
                    vd_status = gr.Textbox(label="Trạng thái", lines=1, interactive=False)

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
                if custom and custom.strip():
                    parts.append(custom.strip())
                return ", ".join(parts) if parts else None

            def _design_fn(text, lang, preset, accent, dialect, custom, ns, gs, dn, sp, du, pp, po):
                instruct = _build_design_instruct(preset, accent, dialect, custom)
                return _gen_core(text, lang, None, instruct, ns, gs, dn, sp, du, pp, po, mode="design")

            vd_btn.click(_design_fn,
                inputs=[vd_text, vd_lang, vd_preset, vd_accent, vd_dialect, vd_custom,
                        vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po],
                outputs=[vd_audio, vd_status])

        # ==========================================================
        # TAB 4: Hướng dẫn
        # ==========================================================
        with gr.TabItem("📖 Hướng dẫn"):
            gr.Markdown("""
### 🎤 Clone giọng nói
1. Nhập văn bản bạn muốn chuyển thành giọng nói
2. (Tuỳ chọn) Tải lên file audio tham chiếu (3-10 giây, rõ ràng)
3. Nhấn **Tạo giọng** và chờ kết quả

**Mẹo:**
- Audio tham chiếu nên rõ, không có tạp âm
- Nên dùng audio cùng ngôn ngữ với văn bản
- Có thể bỏ trống "Văn bản trong audio" để tự nhận dạng
- Không có audio = Mô hình tự chọn giọng

---

### 🎨 Thiết kế giọng nói
1. Nhập văn bản cần nói
2. Chọn loại giọng từ danh sách (Nam/Nữ, Trẻ/Già...)
3. Tùy chọn: thêm giọng Anh hoặc phương ngữ
4. Nhấn **Tạo giọng**

**Mẹo:**
- Chọn preset từ dropdown để dễ sử dụng
- Hoặc tự nhập mô tả **bằng tiếng ANH** ở ô "Tự nhập mô tả giọng"
- **Quan trọng:** Mô hình chỉ chấp nhận mô tả tiếng ANH hoặc tiếng TRUNG

**Ví dụ mô tả tiếng Anh hợp lệ:**
- `female, teenager, high pitch`
- `male, young adult, british accent`
- `child, whisper`

---

### ⚙️ Cài đặt nâng cao
| Tham số | Mặc định | Giải thích |
|---------|----------|------------|
| Tốc độ | 1.0 | >1 nhanh hơn, <1 chậm hơn |
| Thời lượng | Trống | Đặt số giây cố định (bỏ qua tốc độ) |
| Bước suy luận | 32 | Thấp = nhanh, Cao = đẹp hơn |
| CFG | 2.0 | Cao = rõ hơn, Thấp = tự nhiên hơn |

---

*Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice) by Xiaomi AI Lab*
""")

if __name__ == "__main__":
    demo.queue().launch()
