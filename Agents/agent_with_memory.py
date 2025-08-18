import os
from typing import TypedDict, List, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : list[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def process_message(state:AgentState)->AgentState:
    """
    This function processes the message
    """
    response = llm.invoke(state['messages'])
    print(f"\nAI_Response: {response.content}")
    state['messages'].append(AIMessage(content=response.content))
    print("Current state: ", state['messages'])
    return state

graph = StateGraph(AgentState)
graph.add_node("process_message", process_message)
graph.add_edge(START, "process_message")
graph.add_edge("process_message", END)

app = graph.compile()
conversation_history = []

user_input = input("Enter your message: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages": conversation_history})
    conversation_history=result['messages']
    user_input = input("Enter your message: ")

with open("conversation_history.txt", "w") as f:
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"Human: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
print("Conversation history saved to conversation_history.txt")


