from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

class AgentState(TypedDict):
    #Annotated is used to add metadata to the messages(Annotated[datatype, metadata])
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int)->int:
    """
    This function adds two numbers
    """
    return a + b

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]

tools = [add]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def model_call(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content="You are a helpful assistant, answer my questions in best of your capabilities.")
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages":[response]}

def should_continue(state:AgentState)->AgentState:
    """
    This function decides if the agent should continue or not
    """
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "exit"
    else:
        return "continue"
    
graph = StateGraph(AgentState)

graph.add_node("model_call", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "model_call")

graph.add_conditional_edges(
    "model_call",
    should_continue,
    {#Edge:Node
        "exit" : END,
        "continue" : "tools"
    }
)
graph.add_edge("tools", "model_call")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and add 12 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))


