from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage,ToolMessage,BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) #temperature 0 makes model makes more deterministic

#Embedding model must be compatible with the LLM model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",)

pdf_path = "Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"File {pdf_path} not found")

pdf_loader = PyPDFLoader(pdf_path) #loads PDF

try:
    pdf_pages = pdf_loader.load()
    print(f"Loaded {len(pdf_pages)} pages from {pdf_path}")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #chunk_size is the size of the chunk, chunk_overlap is the overlap between chunks


pages_split = text_splitter.split_documents(pdf_pages)
persist_directory = "/Users/purva.samdani/langgraph/Agents"
collection_name = "stock_market_performance"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
try:
    vector_store = Chroma.from_documents(
    documents=pages_split,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name=collection_name
)
    print(f"Vector store created with {len(pages_split)} chunks successfully")
except Exception as e:
    print(f"Error creating vector store: {e}")
    raise

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

@tool
def retrieve_tool(query:str)->str:
    """
    This tool searches and retreives information from the document
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found"
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}: {doc.page_content}")
    return "\n".join(results)

tools = [retrieve_tool]
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def model_call(state:AgentState)->AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state:AgentState):
    """Check if last messaage contains tools calls"""
    messages = state["messages"]
    last_message = messages[-1]
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""
tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools
#LLM agent

def call_llm(state:AgentState)->AgentState:
    """LLM call with current state as conversation history and ability to call tools"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(messages)
    return {"messages": [response]}

# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)

graph.add_node("llm", model_call)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)

graph.add_edge(START, "llm")
graph.add_edge("retriever_agent", "llm")
rag_agent = graph.compile()

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()


