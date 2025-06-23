import streamlit as st
import requests

st.title("Greenway Chatbot Demo")

st.write("Nhập câu hỏi về sản phẩm, danh mục... và nhận câu trả lời bằng tiếng Việt.")

question = st.text_area("Câu hỏi của bạn", "")

if st.button("Gửi câu hỏi"):
    if not question.strip():
        st.warning("Vui lòng nhập câu hỏi.")
    else:
        with st.spinner("Đang lấy câu trả lời..."):
            try:
                # Gọi API FastAPI local
                resp = requests.post(
                    "http://localhost:8000/chat",
                    json={"question": question},
                    timeout=60
                )
                if resp.status_code == 200:
                    answer = resp.json()["answer"]
                    st.success(answer)
                else:
                    st.error(f"Lỗi: {resp.text}")
            except Exception as e:
                st.error(f"Lỗi khi gọi API: {e}")