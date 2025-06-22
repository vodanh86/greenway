# 🤖 Chatbot Sản phẩm sử dụng LangGraph + FAISS + Tools

## ✅ Tính năng

- Hỏi các **danh mục sản phẩm**
- Tìm **danh sách sản phẩm theo tiêu chí**
- **Đề xuất sản phẩm phù hợp**
- Lấy **mô tả và hướng dẫn sử dụng** sản phẩm

## 🛠 Cài đặt

```bash
pip install -r requirements.txt
```

## 🧾 Chuẩn bị dữ liệu

Thêm file `san_pham.xlsx` vào thư mục `data/` với các cột:

- Tên sản phẩm
- Giá
- Màu sắc
- Brand
- Code
- Category
- Mô tả
- Hướng dẫn sử dụng

## 🔑 Thêm API key

Trong file `.env`:

```
OPENAI_API_KEY=your-key-here
```

## 🚀 Chạy chatbot

```bash
python main.py
```

Gõ các câu như:

- "Sản phẩm nào dùng cho nhà bếp dưới 300K?"
- "Có bao nhiêu danh mục sản phẩm?"
- "Gợi ý sản phẩm khử mùi màu xanh"
- "Cho tôi hướng dẫn sử dụng của Khăn rửa bát"
