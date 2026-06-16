import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def chat_node(state: AgentState):
    """The core LLM node."""
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", chat_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)


memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)


app = FastAPI(title="LangGraph Agent API")

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
   
    
    # The config dict passes the thread_id to LangGraph's memory system
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Prepare the input
    input_messages = [HumanMessage(content=request.message)]
    
    # Run the graph
    result = app_graph.invoke({"messages": input_messages}, config=config)
    
    # Extract the final response from the agent
    final_message = result["messages"][-1].content
    
    return {"response": final_message}

@app.get("/")
async def serve_frontend():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "ayush_jaiswal_ui.html")
    
    with open(file_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)