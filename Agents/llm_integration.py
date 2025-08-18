from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()
class AgentState(TypedDict):
    messages : list[HumanMessage]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def process_message(state:AgentState)->AgentState:
    """
    This function processes the message
    """
    response = llm.invoke(state['messages'])
    print(f"\nAI_Response: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process_message", process_message)
graph.add_edge(START, "process_message")
graph.add_edge("process_message", END)

app = graph.compile()

user_input = input("Enter your message: ")
while user_input != "exit":
    app.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter your message: ")