import streamlit as st
import os
import base64
from PIL import Image
import io
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from typing import Annotated
from typing_extensions import TypedDict

# --- Page config ---
st.set_page_config(page_title="Visual Product Analyzer", page_icon="🔍")
st.title("🔍 Visual Product Analyzer")
st.write("Upload a photo of any product and get a full report — price, reviews, and alternatives.")

# --- Session state ---
if "report" not in st.session_state:
    st.session_state.report = None
if "product_name" not in st.session_state:
    st.session_state.product_name = None

# --- Sidebar ---
with st.sidebar:
    st.header("Setup")
    openai_key = st.text_input("OpenAI API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# --- Build Research Agent ---
def build_agent():
    search_tool = TavilySearchResults(
        max_results=5,
        tavily_api_key=os.environ.get("TAVILY_API_KEY", "")
    )
    tools = [search_tool]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
    system_prompt = SystemMessage(content="""
You are a product research assistant. Given a product name:
1. Search for its current price and where to buy it
2. Search for reviews and ratings
3. Search for alternatives and competitors
4. Write a structured report with:
   - Product Overview
   - Price Range
   - Pros and Cons (from reviews)
   - Top Alternatives
   - Verdict
Always search at least 3 times before writing the report.
""")

    def agent_node(state: AgentState):
        messages = [system_prompt] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()

# --- Identify Product from Image ---
def identify_product(image_bytes: bytes) -> str:
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        },
        {
            "type": "text",
            "text": "What product is in this image? Give me the exact product name and model if visible. Be specific and concise — just the product name, nothing else."
        }
    ])
    response = llm.invoke([message])
    return response.content.strip()

# --- Research Product ---
def research_product_stream(product_name: str):
    app = build_agent()
    inputs = {"messages": [HumanMessage(content=f"Research this product thoroughly: {product_name}")]}
    for step in app.stream(inputs, stream_mode="updates"):
        for node_name, update in step.items():
            messages = update.get("messages", [])
            for msg in messages:
                if node_name == "agent":
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            query = tc["args"].get("query", "")
                            yield ("search", f"Searching: {query}")
                    elif hasattr(msg, "content") and msg.content:
                        yield ("report", msg.content)
                elif node_name == "tools":
                    yield ("result", "Search complete, analyzing results...")

# --- Main UI ---
uploaded_file = st.file_uploader("Upload a product photo", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Analyze Product", type="primary"):
        if not openai_key or not tavily_key:
            st.warning("Please enter both API keys in the sidebar.")
        else:
            # Step 1 — Identify product
            with st.spinner("Identifying product from image..."):
                image_bytes = uploaded_file.getvalue()
                product_name = identify_product(image_bytes)
                st.session_state.product_name = product_name

            st.success(f"Product identified: **{product_name}**")

            # Step 2 — Research product
            st.divider()
            st.subheader("Agent Activity")
            final_report = ""

            for update_type, content in research_product_stream(product_name):
                if update_type == "search":
                    st.markdown(f"🔍 {content}")
                elif update_type == "result":
                    st.markdown(f"✅ {content}")
                elif update_type == "report":
                    final_report = content

            if final_report:
                st.divider()
                st.subheader(f"Product Report: {product_name}")
                st.markdown(final_report)
                st.session_state.report = final_report

                st.download_button(
                    label="Download Report",
                    data=final_report,
                    file_name=f"{product_name[:30].replace(' ', '_')}_report.txt",
                    mime="text/plain"
                )
