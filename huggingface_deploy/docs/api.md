# OmniVoice API Documentation

Chuyển văn bản thành giọng nói (TTS) qua REST API.

## Base URL

```
https://daodat242-omnivoice-tieng-viet.hf.space
```

## Authentication

Không cần. API hoàn toàn miễn phí và công khai.

## Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/tts` | Chuyển văn bản thành giọng nói |
| `POST` | `/api/clone` | Clone giọng nói từ audio tham chiếu |
| `POST` | `/api/design` | Thiết kế giọng nói tùy chỉnh |

---

## 1. POST `/api/tts`

Chuyển văn bản thành giọng nói đa ngôn ngữ.

### Request Body

```json
{
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
}
```

### Tham số

| # | Tên | Kiểu | Bắt buộc | Mặc định | Mô tả |
|---|-----|------|----------|----------|-------|
| 0 | `text` | string | ✅ | - | Văn bản cần chuyển thành giọng nói |
| 1 | `language` | string | ❌ | `"Auto"` | Ngôn ngữ đầu ra (xem danh sách ngôn ngữ) |
| 2 | `voice_type` | string | ❌ | `"Tự động"` | Loại giọng (xem bảng giọng bên dưới) |
| 3 | `pitch` | string | ❌ | `"Không chọn"` | Âm vực: `Cực thấp`, `Thấp`, `Trung bình`, `Cao`, `Cực cao` |
| 4 | `volume` | float | ❌ | `1.0` | Âm lượng (0.1 - 3.0) |
| 5 | `speed` | float | ❌ | `1.0` | Tốc độ nói (0.5 - 1.5) |
| 6 | `duration` | float | ❌ | `null` | Thời lượng cố định (giây). Nếu set sẽ bỏ qua speed |
| 7 | `num_step` | int | ❌ | `32` | Bước suy luận (4-64). Cao = đẹp hơn nhưng chậm hơn |
| 8 | `guidance_scale` | float | ❌ | `2.0` | CFG scale (0.0-4.0). Cao = rõ hơn, thấp = tự nhiên hơn |

### Response

```json
{
  "data": [
    {
      "is_file": false,
      "name": "/tmp/gradio/xxx/audio.wav",
      "data": null,
      "size": null,
      "orig_name": "audio.wav",
      "mime_type": "audio/wav",
      "is_stream": false,
      "alt_text": null
    },
    "✅ Hoàn tất! Nhấn ▶ để nghe"
  ]
}
```

### Ví dụ đầy đủ

#### Python (gradio_client)

```bash
pip install gradio_client
```

```python
from gradio_client import Client

client = Client("Daodat242/omnivoice-tieng-viet")

# TTS tiếng Việt
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

# result[0] = (sample_rate, numpy_array)
# result[1] = status message
sample_rate, audio_data = result[0]
print(f"Sample rate: {sample_rate}, Length: {len(audio_data)}")
```

#### Python (requests)

```python
import requests

url = "https://daodat242-omnivoice-tieng-viet.hf.space/api/tts"
payload = {
    "data": [
        "Hello, how are you?",
        "English",
        "Nam - Trẻ tuổi",
        "Không chọn",
        1.0,
        1.0,
        None,
        32,
        2.0
    ]
}

response = requests.post(url, json=payload)
result = response.json()
print(result)
```

#### cURL

```bash
curl -X POST "https://daodat242-omnivoice-tieng-viet.hf.space/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      "Xin chào!",
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

#### JavaScript (Node.js)

```javascript
const response = await fetch(
  "https://daodat242-omnivoice-tieng-viet.hf.space/api/tts",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data: [
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
    })
  }
);

const result = await response.json();
console.log(result);
```

---

## 2. POST `/api/clone`

Clone giọng nói từ file audio tham chiếu.

### Request Body

```json
{
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
}
```

### Tham số

| # | Tên | Kiểu | Bắt buộc | Mặc định | Mô tả |
|---|-----|------|----------|----------|-------|
| 0 | `text` | string | ✅ | - | Văn bản cần nói |
| 1 | `language` | string | ❌ | `"Auto"` | Ngôn ngữ đầu ra |
| 2 | `ref_audio` | file | ❌ | `null` | File audio tham chiếu (3-10 giây, WAV/MP3) |
| 3 | `ref_text` | string | ❌ | `null` | Nội dung trong audio (để trống = tự nhận dạng AI) |
| 4 | `num_step` | int | ❌ | `32` | Bước suy luận |
| 5 | `guidance_scale` | float | ❌ | `2.0` | CFG scale |
| 6 | `denoise` | bool | ❌ | `true` | Khử nhiễu audio |
| 7 | `speed` | float | ❌ | `1.0` | Tốc độ nói |
| 8 | `duration` | float | ❌ | `null` | Thời lượng cố định (giây) |
| 9 | `preprocess_prompt` | bool | ❌ | `true` | Tiền xử lý text |
| 10 | `postprocess_output` | bool | ❌ | `true` | Xử lý audio đầu ra |
| 11 | `instruct` | string | ❌ | `null` | Chỉ dẫn giọng nói (tiếng ANH) |

### Ví dụ

```python
from gradio_client import Client

client = Client("Daodat242/omnivoice-tieng-viet")

# Clone giọng với audio tham chiếu
result = client.predict(
    text="Xin chào!",
    language="Tiếng Việt",
    ref_audio="path/to/reference.wav",  # Upload audio file
    ref_text="Nội dung trong audio",     # hoặc để trống
    num_step=32,
    guidance_scale=2.0,
    denoise=True,
    speed=1.0,
    duration=None,
    preprocess_prompt=True,
    postprocess_output=True,
    instruct=None,
    api_name="/clone"
)
```

---

## 3. POST `/api/design`

Thiết kế giọng nói tùy chỉnh.

### Request Body

```json
{
  "data": [
    "Xin chào!",
    "Tiếng Việt",
    "Nữ - Trẻ tuổi",
    "Giọng Mỹ",
    "Không chọn",
    "",
    32,
    2.0,
    true,
    1.0,
    null,
    true,
    true
  ]
}
```

### Tham số

| # | Tên | Kiểu | Bắt buộc | Mặc định | Mô tả |
|---|-----|------|----------|----------|-------|
| 0 | `text` | string | ✅ | - | Văn bản cần nói |
| 1 | `language` | string | ❌ | `"Auto"` | Ngôn ngữ đầu ra |
| 2 | `preset` | string | ❌ | `"Không chọn"` | Loại giọng (xem bảng giọng) |
| 3 | `accent` | string | ❌ | `"Không chọn"` | Giọng regional |
| 4 | `dialect` | string | ❌ | `"Không chọn"` | Phương ngữ (tiếng Trung) |
| 5 | `custom` | string | ❌ | `""` | Mô tả tùy chỉnh (tiếng ANH) |
| 6 | `num_step` | int | ❌ | `32` | Bước suy luận |
| 7 | `guidance_scale` | float | ❌ | `2.0` | CFG scale |
| 8 | `denoise` | bool | ❌ | `true` | Khử nhiễu |
| 9 | `speed` | float | ❌ | `1.0` | Tốc độ nói |
| 10 | `duration` | float | ❌ | `null` | Thời lượng cố định |
| 11 | `preprocess_prompt` | bool | ❌ | `true` | Tiền xử lý |
| 12 | `postprocess_output` | bool | ❌ | `true` | Xử lý đầu ra |

### Bảng giọng (Preset)

| Giá trị tiếng Việt | Giá trị API (English) |
|--------------------|-----------------------|
| Nam - Trung niên | `male, middle-aged, moderate pitch` |
| Nam - Trẻ tuổi | `male, young adult, moderate pitch` |
| Nam - Già | `male, elderly, low pitch` |
| Nam - Thiếu niên | `male, teenager, high pitch` |
| Nữ - Trung niên | `female, middle-aged, moderate pitch` |
| Nữ - Trẻ tuổi | `female, young adult, moderate pitch` |
| Nữ - Già | `female, elderly, low pitch` |
| Nữ - Thiếu niên | `female, teenager, high pitch` |
| Trẻ em | `child, moderate pitch` |
| Thì thầm (Nam) | `male, whisper` |
| Thì thầm (Nữ) | `female, whisper` |

### Giọng Regional (Accent)

| Giá trị | Mô tả |
|---------|-------|
| `american accent` | Giọng Mỹ |
| `british accent` | Giọng Anh |
| `australian accent` | Giọng Úc |
| `indian accent` | Giọng Ấn Độ |
| `japanese accent` | Giọng Nhật |
| `korean accent` | Giọng Hàn |
| `russian accent` | Giọng Nga |
| `chinese accent` | Giọng Trung |

---

## Danh sách ngôn ngữ phổ biến

| Mã | Ngôn ngữ | Tên hiển thị trong API |
|----|----------|------------------------|
| `vi` | Tiếng Việt | `"Tiếng Việt"` |
| `en` | English | `"English (Tiếng Anh)"` |
| `zh` | 中文 | `"中文 (Tiếng Trung)"` |
| `ja` | 日本語 | `"日本語 (Tiếng Nhật)"` |
| `ko` | 한국어 | `"한국어 (Tiếng Hàn)"` |
| `fr` | Français | `"Français (Tiếng Pháp)"` |
| `de` | Deutsch | `"Deutsch (Tiếng Đức)"` |
| `es` | Español | `"Español (Tiếng Tây Ban Nha)"` |
| `th` | ไทย | `"ไทย (Tiếng Thái)"` |
| `id` | Bahasa Indonesia | `"Bahasa Indonesia (Tiếng Indonesia)"` |
| `pt` | Português | `"Português (Tiếng Bồ Đào Nha)"` |
| `it` | Italiano | `"Italiano (Tiếng Ý)"` |
| `ru` | Русский | `"Русский (Tiếng Nga)"` |

> **Lưu ý:** Truyền `null` hoặc `"Auto"` để mô hình tự nhận diện ngôn ngữ.

---

## Xử lý lỗi

### Response khi lỗi

```json
{
  "error": "Lỗi: ValueError: Text cannot be empty"
}
```

### Các lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|------|-------------|-----------|
| `Text cannot be empty` | `text` rỗng | Nhập văn bản |
| `Invalid language` | Ngôn ngữ không hỗ trợ | Dùng mã ngôn ngữ hợp lệ |
| `File too large` | Audio > 20 giây | Cắt audio < 20 giây |
| `Timeout` | Server quá tải | Thử lại sau |

---

## Giới hạn

| Hạng mục | Giới hạn |
|----------|----------|
| Độ dài text | < 100 từ (tối ưu), 100-200 (chấp nhận), > 300 (không khuyến khích) |
| Audio tham chiếu | 3-10 giây (tối ưu), tối đa 20 giây (tự cắt) |
| Tốc độ | 5-30 giây/request tùy độ dài |
| Rate limit | ~60 requests/phút (Gradio default) |
| Chi phí | Miễn phí |

---

## Ví dụ tích hợp

### Discord Bot (Python)

```python
import discord
from gradio_client import Client

bot = discord.Client(intents=discord.Intents.default())
tts_client = Client("Daodat242/omnivoice-tieng-viet")

@bot.command()
async def say(ctx, *, text: str):
    result = tts_client.predict(
        text=text,
        language="Tiếng Việt",
        api_name="/tts"
    )
    # result[0] = (sample_rate, numpy_array)
    # Convert to audio file and send
    import numpy as np
    import soundfile as sf
    sr, audio = result[0]
    sf.write("output.wav", audio, sr)
    await ctx.send(file=discord.File("output.wav"))

bot.run("YOUR_BOT_TOKEN")
```

### Web App (JavaScript)

```javascript
async function textToSpeech(text) {
  const response = await fetch(
    "https://daodat242-omnivoice-tieng-viet.hf.space/api/tts",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data: [text, "Tiếng Việt", "Tự động", "Không chọn", 1.0, 1.0, null, 32, 2.0]
      })
    }
  );

  const result = await response.json();
  // Play audio from result.data[0]
  const audio = new Audio(result.data[0].url);
  audio.play();
}
```

---

## Liên hệ

- GitHub: [Daodat242/omnivoice-tieng-viet](https://github.com/Daodat242/omnivoice-tieng-viet)
- HuggingFace: [Daodat242/omnivoice-tieng-viet](https://huggingface.co/spaces/Daodat242/omnivoice-tieng-viet)
- Issues: [GitHub Issues](https://github.com/Daodat242/omnivoice-tieng-viet/issues)
