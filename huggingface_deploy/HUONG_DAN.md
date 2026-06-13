# Huong dan deploy len HuggingFace Spaces

## Buoc 1: Tao tai khoan HuggingFace
1. Vao https://huggingface.co
2. Nhan "Sign Up" de tao tai khoan mien phi
3. Xac minh email

## Buoc 2: Tao Space moi
1. Dang nhap vao HuggingFace
2. Nhan vao avatar (goc tren phai) -> "New Space"
3. Dien thong tin:
   - Space name: `omnivoice-tieng-viet` (hoac ten khac)
   - License: `Apache 2.0`
   - SDK: `Gradio`
   - Hardware: `CPU Basic` (mien phi) hoac `T4 small` (neu co)
4. Nhan "Create Space"

## Buoc 3: Upload file
1. Vao Space vua tao
2. Nhan tab "Files"
3. Nhan "Add file" -> "Upload files"
4. Upload 3 file:
   - `app.py` (tu thu muc huggingface_deploy)
   - `requirements.txt` (tu thu muc huggingface_deploy)
   - `README.md` (tu thu muc huggingface_deploy)

## Buoc 4: Cho Space build
1. Sau khi upload xong, Space se tu dong build
2. Qua trinh build mat khoang 5-10 phut
3. Khi thanh cong, ban se thay giao dien Gradio

## Buoc 5: Su dung
1. Nhap van ban tieng Viet
2. Chon giong noi
3. Nhan "Tao giong"
4. Tai ket qua ve may

## Luu y
- Space mien phi se cham (5-20s moi lan generate)
- Van ban ngan (< 50 tu) se nhanh hon
- Neu muon nhanh hon, can upgrade len GPU ($0.60/gio)

## Fix loi (neu co)
Nei Space bi loi, kiem tra:
1. Tab "Logs" de xem loi chi tiet
2. Danh may dung SDK va version
3. Kiem tra file requirements.txt dung format
