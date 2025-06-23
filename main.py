from typing_extensions import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import os
import pandas as pd
from fastapi import FastAPI
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load data
categories = {}
xl = pd.ExcelFile("subsections_xlsxwriter.xlsx")
for sheet in xl.sheet_names:
    categories[sheet.lower()] = sheet

embedding = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "faiss_index", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

SYSTEM_PROMPT = "Luôn trả lời bằng tiếng Việt, không quá 500 chữ."

# --- ROUTING SCHEMA ---
class Route(BaseModel):
    step: Literal[
        "list_categories",
        "get_products_by_category",
        "search_products",
        "suggest_products",
        "get_production_info"
    ] = Field(None, description="Bước tiếp theo trong quá trình routing")
    category: str = Field(None, description="Tên danh mục sản phẩm nếu có")
    product_name: str = Field(None, description="Tên sản phẩm nếu có")

router = llm.with_structured_output(Route)

# --- STATE ---
class State(TypedDict):
    input: str
    decision: str
    output: str
    category: str
    product_name: str

# --- NODES ---
def node_list_categories(state: State):
    with open("docs/thong_tin_danh_muc.txt", "r", encoding="utf-8") as f:
        result = f.read()
    return {"output": result}

def node_get_products_by_category(state: State):
    """Trả về danh sách sản phẩm theo tên category (danh mục)"""
    input_category = state.get("category") or state["input"]
    input_lower = input_category.lower().strip()
    print(f"Input category: {input_lower}")
    print(f"Available categories: {categories.keys()}")
    try:
        if input_lower not in categories:
            return {
                "output": (
                    f"Không tìm thấy danh mục (sheet) '{input_category}' trong file Excel.\n"
                    f"Các danh mục hiện có: {', '.join(categories.keys())}"
                )
            }
        df_cat = xl.parse(sheet_name=categories[input_lower])
    except Exception as e:
        return {"output": f"Lỗi khi đọc file: {e}"}
    if df_cat.empty:
        return {"output": f"Không có sản phẩm nào trong danh mục '{input_category}'."}
    lines = []
    for _, row in df_cat.iterrows():
        name = row.get('Tên sản phẩm', 'Không rõ')
        price = row.get('Giá', 'Không rõ')
        brand = row.get('Thương hiệu', 'Không có')
        lines.append(f"- {name} | Giá: {price} | thương hiệu: {brand}")
    return {"output": "\n".join(lines)}

def node_search_products(state: State):
    matches = retriever.get_relevant_documents(state["input"])
    print(f"Found {len(matches)} matches for search query: {state['input']}")
    if not matches:
        return {"output": "Không tìm thấy sản phẩm nào phù hợp."}
    return {"output": "\n\n".join([doc.page_content for doc in matches[:5]])}

def node_suggest_products(state: State):
    matches = retriever.get_relevant_documents("gợi ý " + state["input"])
    if not matches:
        return {"output": "Không có gợi ý phù hợp."}
    return {"output": "\n\n".join([doc.page_content for doc in matches[:3]])}

def node_get_production_info(state: State):
    """Tìm thông tin sản phẩm gần đúng trong FAISS (semantic search)."""
    query = state.get("product_name") or state["input"]
    matches = retriever.get_relevant_documents(query)
    product_details = [doc for doc in matches if doc.metadata.get("type") == "product_detail"]
    if product_details:
        return {"output": "\n\n".join(doc.page_content for doc in product_details[:3])}
    suggestions = [
        doc.page_content for doc in matches if doc.metadata.get("type") == "product_name"
    ]
    if suggestions:
        return {
            "output": (
                "Không tìm thấy thông tin chi tiết, bạn có muốn hỏi về một trong các sản phẩm sau không?\n- "
                + "\n- ".join(suggestions[:5])
            )
        }
    return {"output": "Không tìm thấy sản phẩm nào phù hợp."}

def node_router(state: State):
    """Route input đến node phù hợp"""
    decision = router.invoke(
        [
            SystemMessage(
                content=SYSTEM_PROMPT + "\nRoute truy vấn của người dùng đến một trong các bước: list_categories, get_products_by_category, search_products, suggest_products, get_production_info dựa trên ý định. Nếu có, hãy trích xuất cả tên danh mục (category) và tên sản phẩm (product_name)."
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    print(f"Decision made: {decision.step}, category: {getattr(decision, 'category', None)}, product_name: {getattr(decision, 'product_name', None)}")
    return {
        "decision": decision.step,
        "category": getattr(decision, "category", None),
        "product_name": getattr(decision, "product_name", None)
    }

# --- ROUTING LOGIC ---
def route_decision(state: State):
    if state["decision"] == "list_categories":
        return "node_list_categories"
    elif state["decision"] == "get_products_by_category":
        return "node_get_products_by_category"
    elif state["decision"] == "search_products":
        return "node_search_products"
    elif state["decision"] == "suggest_products":
        return "node_suggest_products"
    elif state["decision"] == "get_production_info":
        return "node_get_production_info"

def node_llm_postprocess(state: State):
    """Xử lý kết quả đầu ra bằng LLM trước khi trả về, sử dụng context là kết quả truy vấn và input là câu hỏi gốc."""
    print(f"Postprocessing output with context: {state['output']}")
    messages = [
        SystemMessage(
            content=(
                "Bạn hãy dựa vào phần context dưới đây để trả lời câu hỏi của người dùng. "
                "Nếu context không liên quan hoặc không đủ thông tin, hãy trả lời lịch sự rằng bạn không tìm thấy kết quả phù hợp. "
                "Đảm bảo trả lời bằng tiếng Việt, không quá 500 chữ.\n\n"
                f"Context:\n{state['output']}"
            )
        ),
        HumanMessage(content=state["input"])
    ]
    result = llm.invoke(messages)
    return {"output": result.content}

# --- BUILD WORKFLOW ---
router_builder = StateGraph(State)
router_builder.add_node("node_list_categories", node_list_categories)
router_builder.add_node("node_get_products_by_category", node_get_products_by_category)
router_builder.add_node("node_search_products", node_search_products)
router_builder.add_node("node_suggest_products", node_suggest_products)
router_builder.add_node("node_get_production_info", node_get_production_info)
router_builder.add_node("node_router", node_router)
router_builder.add_node("node_llm_postprocess", node_llm_postprocess)

router_builder.add_edge(START, "node_router")
router_builder.add_conditional_edges(
    "node_router",
    route_decision,
    {
        "node_list_categories": "node_list_categories",
        "node_get_products_by_category": "node_get_products_by_category",
        "node_search_products": "node_search_products",
        "node_suggest_products": "node_suggest_products",
        "node_get_production_info": "node_get_production_info",
    },
)
# Kết nối các node kết quả về node_llm_postprocess thay vì END
router_builder.add_edge("node_list_categories", "node_llm_postprocess")
router_builder.add_edge("node_get_products_by_category", "node_llm_postprocess")
router_builder.add_edge("node_search_products", "node_llm_postprocess")
router_builder.add_edge("node_suggest_products", "node_llm_postprocess")
router_builder.add_edge("node_get_production_info", "node_llm_postprocess")
router_builder.add_edge("node_llm_postprocess", END)

router_workflow = router_builder.compile()

# --- FASTAPI ---
app = FastAPI()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    state = router_workflow.invoke({"input": req.question})
    return ChatResponse(answer=state["output"])