from typing import TypedDict, Annotated, Sequence, Optional
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import re
import json
from datetime import datetime

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

class AgentState(TypedDict):
    name: Optional[str]
    company: Optional[str]
    designation: Optional[str]
    email: Optional[str]
    llm_suggestion: Optional[str]
    json_output: Optional[str]     

def get_name(state: AgentState) -> AgentState:
    """
    This function gets the name of the client
    """
    print("AI: What is your name?")
    user_input = input("User: ")
    state["name"] = user_input
    return state

def get_company(state: AgentState) -> AgentState:
    """
    This function gets the company of the client
    """
    print("AI: What is your company?")
    user_input = input("User: ")
    state["company"] = user_input
    return state

def get_designation(state: AgentState) -> AgentState:
    """
    This function gets the designation of the client
    """
    print("AI: What is your designation?")
    user_input = input("User: ")
    state["designation"] = user_input
    return state

def get_email(state: AgentState) -> AgentState:
    """
    This function gets the email of the client
    """     
    print("AI: What is your email?")
    user_input = input("User: ")
    state["email"] = user_input
    return state

def suggest_name_correction(state: AgentState) -> AgentState:
    """
    This function uses LLM to suggest name corrections
    """
    if state["name"]:
        prompt = f"""
        The user provided this name: "{state["name"]}"
        
        This name seems too short or invalid. Please provide a helpful suggestion to the user.
        Ask them to provide their full name or correct any obvious issues.
        
        Respond in a friendly, conversational way starting with "AI: "
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        state["llm_suggestion"] = response.content
        print(response.content)
    else:
        print("AI: Please provide your name. It cannot be empty.")
    
    return state

def suggest_company_correction(state: AgentState) -> AgentState:
    """
    This function uses LLM to suggest company corrections
    """
    if state["company"]:
        prompt = f"""
        The user provided this company name: "{state["company"]}"
        
        This company name seems too short or invalid. Please provide a helpful suggestion to the user.
        Ask them to provide the full company name or correct any obvious issues.
        
        Respond in a friendly, conversational way starting with "AI: "
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        state["llm_suggestion"] = response.content
        print(response.content)
    else:
        print("AI: Please provide your company name. It cannot be empty.")
    
    return state

def suggest_designation_correction(state: AgentState) -> AgentState:
    """
    This function uses LLM to suggest designation corrections
    """
    if state["designation"]:
        prompt = f"""
        The user provided this designation: "{state["designation"]}"
        
        This designation seems too short or invalid. Please provide a helpful suggestion to the user.
        Ask them to provide their full job title/designation or correct any obvious issues.
        
        Respond in a friendly, conversational way starting with "AI: "
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        state["llm_suggestion"] = response.content
        print(response.content)
    else:
        print("AI: Please provide your designation. It cannot be empty.")
    
    return state

def suggest_email_correction(state: AgentState) -> AgentState:
    """
    This function uses LLM to suggest email corrections
    """
    if state["email"]:
        prompt = f"""
        The user provided this email: "{state["email"]}"
        
        This email appears to be invalid or incomplete. Common issues include:
        - Missing @ symbol
        - Missing domain extension (like .com, .org, etc.)
        - Typos in common domains (gmail, yahoo, outlook, etc.)
        
        Please analyze the email and suggest a corrected version. For example:
        - If they wrote "john@gmailcom" suggest "john@gmail.com"
        - If they wrote "sarah@yahoo" suggest "sarah@yahoo.com"
        - If they wrote "mike.company" suggest "mike@company.com"
        
        Respond in a friendly way starting with "AI: " and ask something like "Did you mean [corrected_email]?"
        
        Only suggest corrections for obvious issues. If the email is completely garbled, ask them to re-enter it.
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        state["llm_suggestion"] = response.content
        print(response.content)
    else:
        print("AI: Please provide your email address. It cannot be empty.")
    
    return state

# Validation nodes now use lambda functions since they just pass through state

# Routing functions for conditional edges
def route_after_name_validation(state: AgentState) -> str:
    """Determine where to go after name validation"""
    if state["name"] and len(state["name"].strip()) > 2:
        return "continue"
    return "llm_suggest"

def route_after_company_validation(state: AgentState) -> str:
    """Determine where to go after company validation"""
    if state["company"] and len(state["company"].strip()) > 2:
        return "continue"
    return "llm_suggest"

def route_after_designation_validation(state: AgentState) -> str:
    """Determine where to go after designation validation"""
    if state["designation"] and len(state["designation"].strip()) > 2:
        return "continue"
    return "llm_suggest"

def route_after_email_validation(state: AgentState) -> str:
    """Determine where to go after email validation"""
    if state["email"]:
        email = state["email"].strip()
        # Basic email validation with regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, email):
            return "create_json_output"
    return "llm_suggest"

def create_json_output(state: AgentState) -> AgentState:
    """
    This function creates a JSON output of all collected client data
    """
    client_data = {
        "client_details": {
            "name": state["name"],
            "company": state["company"],
            "designation": state["designation"],
            "email": state["email"]
        },
        "collection_metadata": {
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
    }
    
    # Create formatted JSON string
    json_output = json.dumps(client_data, indent=2)
    state["json_output"] = json_output
    
    return state

# Create the state graph
agent_graph = StateGraph(AgentState)

# Add all nodes
agent_graph.add_node("get_name", get_name)
agent_graph.add_node("get_company", get_company)
agent_graph.add_node("get_designation", get_designation)
agent_graph.add_node("get_email", get_email)
agent_graph.add_node("validate_name", lambda state: state)
agent_graph.add_node("validate_company", lambda state: state)
agent_graph.add_node("validate_designation", lambda state: state)
agent_graph.add_node("validate_email", lambda state: state)
agent_graph.add_node("create_json_output", create_json_output)

# Add LLM suggestion nodes
agent_graph.add_node("suggest_name_correction", suggest_name_correction)
agent_graph.add_node("suggest_company_correction", suggest_company_correction)
agent_graph.add_node("suggest_designation_correction", suggest_designation_correction)
agent_graph.add_node("suggest_email_correction", suggest_email_correction)

# Add edges and conditional logic
agent_graph.add_edge(START, "get_name")
agent_graph.add_edge("get_name", "validate_name")
agent_graph.add_conditional_edges(
    "validate_name",
    route_after_name_validation,
    {
        "continue": "get_company",
        "llm_suggest": "suggest_name_correction"
    }
)
agent_graph.add_edge("suggest_name_correction", "get_name")

agent_graph.add_edge("get_company", "validate_company")
agent_graph.add_conditional_edges(
    "validate_company",
    route_after_company_validation,
    {
        "continue": "get_designation",
        "llm_suggest": "suggest_company_correction"
    }
)
agent_graph.add_edge("suggest_company_correction", "get_company")

agent_graph.add_edge("get_designation", "validate_designation")
agent_graph.add_conditional_edges(
    "validate_designation",
    route_after_designation_validation,
    {
        "continue": "get_email",
        "llm_suggest": "suggest_designation_correction"
    }
)
agent_graph.add_edge("suggest_designation_correction", "get_designation")

agent_graph.add_edge("get_email", "validate_email")
agent_graph.add_conditional_edges(
    "validate_email",
    route_after_email_validation,
    {
        "create_json_output": "create_json_output",
        "llm_suggest": "suggest_email_correction"
    }
)
agent_graph.add_edge("suggest_email_correction", "get_email")

# Add final edge from JSON output to END
agent_graph.add_edge("create_json_output", END)

# Compile the graph
compiled_graph = agent_graph.compile()

def run_client_details_collection():
    """
    Run the client details collection system
    """
    print("🎯 Welcome to the Client Details Collection System!")
    print("This AI-powered system will help you enter your information correctly.\n")
    
    # Initial state - all fields start as None
    initial_state = {
        "name": None,
        "company": None,
        "designation": None,
        "email": None,
        "llm_suggestion": None,
        "json_output": None
    }
    
    final_state = initial_state  # Initialize final_state
    
    try:
        # Run the graph
        final_state = compiled_graph.invoke(initial_state)
        
        print("\n✅ Client details collection completed successfully!")
        print("Final collected data:")
        if final_state.get("json_output"):
            print(final_state["json_output"])
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Please check your inputs and try again.")
    
    return final_state

# Run the system if this file is executed directly
if __name__ == "__main__":
    run_client_details_collection()