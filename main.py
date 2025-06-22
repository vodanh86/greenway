from langgraph.graph import StateGraph
import os
import pandas as pd
from fastapi import FastAPI
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.agents import tool, initialize_agent, AgentType
from langchain.tools.render import render_text_description
from pydantic import BaseModel
from typing import Optional

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
docs = []
categories = {}

xl = pd.ExcelFile("subsections_xlsxwriter.xlsx")
for sheet in xl.sheet_names:
    # Lưu category
    categories[sheet.lower()] = sheet
    df_cat = xl.parse(sheet_name=sheet)
    for _, row in df_cat.iterrows():
        name = row.get('Tên sản phẩm', '')
        # Lưu product name
        docs.append(Document(
            page_content=f"Product name: {name}",
            metadata={"type": "product_name"}
        ))
        # Lưu product detail
        detail = f"""Tên sản phẩm: {name}
Danh mục: {sheet}
Giá: {row.get('Giá', '')}
Khối lượng: {row.get('Khối lượng', '')}
Mô tả: {row.get('Mô tả', '')}
Mô tả đủ: {row.get('Mô tả đủ', '')}
Hướng dẫn sử dụng: {row.get('Hướng dẫn sử dụng', '')}
Thành phần: {row.get('Thành phần', '')}
Thương hiệu: {row.get('Thương hiệu', '')}
Hình ảnh: {row.get('Hình ảnh', '')}
"""
        docs.append(Document(
            page_content=detail,
            metadata={
                "type": "product_detail",
            }
        ))


for cat in categories.keys():
    docs.append(Document(
        page_content=f"Category: {cat}",
        metadata={"type": "category"}
    ))

if False:
    # Tạo FAISS với metadata
    embedding = OpenAIEmbeddings()
    batch_size = 100  # hoặc 1000 tuỳ lượng data
    vectorstore = None

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embedding)
        else:
            vectorstore.add_documents(batch)

    vectorstore.save_local("faiss_index")
    retriever = vectorstore.as_retriever()
    # Định nghĩa tools

embedding = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "faiss_index", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()


@tool
def list_categories(input: str) -> str:
    """Liệt kê các danh mục sản phẩm chi theo 2 cấp."""
    return open("docs/thong_tin_danh_muc.txt", "r", encoding="utf-8").read()


@tool
def get_products_by_category(input: str) -> str:
    """Trả về danh sách sản phẩm theo tên category (danh mục), đọc từ sheet cùng tên trong file subsections_xlsm."""
    import pandas as pd
    try:
        # Lấy danh sách sheet
        if input.lower() not in categories:
            return (
                f"Không tìm thấy danh mục (sheet) '{input}' trong file Excel.\n"
                f"Các danh mục hiện có: {', '.join(categories.keys())}"
            )
        # Đọc sheet theo tên category (input)
        df_cat = xl.parse(sheet_name=categories[input.lower()])
    except Exception as e:
        return f"Lỗi khi đọc file: {e}"
    if df_cat.empty:
        return f"Không có sản phẩm nào trong danh mục '{input}'."
    lines = []
    for _, row in df_cat.iterrows():
        name = row.get('Tên sản phẩm', 'Không rõ')
        price = row.get('Giá', 'Không rõ')
        brand = row.get('Thương hiệu', 'Không có')
        lines.append(f"- {name} | Giá: {price} | thương hiệu: {brand}")
    print(f"Đã tìm thấy {len(lines)} sản phẩm trong danh mục '{input}'.")
    return "\n".join(lines)


@tool
def search_products(input: str) -> str:
    """Tìm kiếm sản phẩm theo từ khóa."""
    matches = retriever.get_relevant_documents(input)
    return "\n\n".join([doc.page_content for doc in matches[:5]])


@tool
def suggest_products(input: str) -> str:
    """Gợi ý sản phẩm dựa trên từ khóa."""
    matches = retriever.get_relevant_documents("gợi ý " + input)
    return "\n\n".join([doc.page_content for doc in matches[:3]])


@tool
def get_production_info(input: str) -> str:
    """Tìm thông tin sản phẩm gần đúng trong FAISS (semantic search)."""
    matches = retriever.get_relevant_documents(input)
    # Lọc chỉ các document có type là product_detail
    product_details = [doc for doc in matches if doc.metadata.get("type") == "product_detail"]
    if product_details:
        return "\n\n".join(doc.page_content for doc in product_details[:3])
    # Nếu không có, trả về gợi ý tên sản phẩm gần đúng (nếu có)
    suggestions = [
        doc.page_content for doc in matches if doc.metadata.get("type") == "product_name"
    ]
    if suggestions:
        return (
            "Không tìm thấy thông tin chi tiết, bạn có muốn hỏi về một trong các sản phẩm sau không?\n- "
            + "\n- ".join(suggestions[:5])
        )
    return "Không tìm thấy sản phẩm nào phù hợp."


# Kết hợp tools
tools = [list_categories, get_products_by_category,
         search_products, suggest_products, get_production_info]

# Tạo LLM hỗ trợ tool
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Tạo agent executor để LLM thực thi tool thực sự
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)


class ChatState(BaseModel):
    input: str
    result: Optional[str] = None


def process_node(state):
    question = "Trả lời bằng tiếng Việt, không quá 500 chữ. " + state.input
    print(f"\n[Agent] Nhận câu hỏi: {question}")
    result = agent.run(question)
    print(f"[Agent] Kết quả Agent: {result}")
    return {"result": result}


# Nếu bạn vẫn muốn giữ LangGraph flow:

builder = StateGraph(state_schema=ChatState)
builder.add_node("chat_with_tools", process_node)
builder.set_entry_point("chat_with_tools")
graph = builder.compile()

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    output = graph.invoke({"input": req.question})
    return ChatResponse(answer=output["result"])
