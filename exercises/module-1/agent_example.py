from typing import TypedDict, Annotated, Sequence, Union
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import datetime
import os

# Load environment variables
load_dotenv('../.env')

# Define our tools
@tool()
def get_current_time() -> str:
    """Get the current time"""
    return datetime.datetime.now().strftime("%H:%M:%S")

@tool()
def calculate(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    next_step: str

def create_agent():
    """Create an agent that can use tools"""
    # Initialize our chat model
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    
    # List of available tools
    tools = [get_current_time, calculate]
    
    def should_continue(state: AgentState) -> Union[bool, str]:
        """Determine if we should continue or end"""
        last_message = state["messages"][-1]
        if "FINAL ANSWER:" in last_message.content:
            return END
        return "agent"
    
    def agent_function(state: AgentState) -> AgentState:
        """Process the messages and use tools if needed"""
        messages = state["messages"]
        
        # Add system message about available tools
        system_message = f"""You are a helpful AI assistant with access to the following tools:
        {[tool.name for tool in tools]}
        
        To use a tool, respond with:
        TOOL: <tool_name>
        ARGS: <tool_args>
        
        When you have a final answer, respond with:
        FINAL ANSWER: <your response>
        """
        
        # Get response from the chat model
        response = chat.invoke([*messages, AIMessage(content=system_message)])
        
        # Check if the response indicates tool use
        if "TOOL:" in response.content:
            # Parse tool name and args
            tool_name = response.content.split("TOOL:")[1].split("\n")[0].strip()
            tool_args = response.content.split("ARGS:")[1].split("\n")[0].strip()
            
            # Find and use the tool
            for tool in tools:
                if tool.name == tool_name:
                    tool_result = tool(tool_args)
                    return {
                        "messages": [
                            *messages,
                            AIMessage(content=f"Using tool {tool_name}..."),
                            AIMessage(content=f"Tool result: {tool_result}")
                        ],
                        "next_step": "agent"
                    }
        
        return {
            "messages": [*messages, response],
            "next_step": "agent"
        }
    
    # Create our graph
    workflow = StateGraph(AgentState)
    
    # Add our node
    workflow.add_node("agent", agent_function)
    
    # Add our edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue
    )
    
    # Compile the graph
    chain = workflow.compile()
    
    return chain

def run_example():
    """Run an example of our agent"""
    # Create the agent
    agent = create_agent()
    
    # Create initial state
    messages = [
        HumanMessage(content="What is the current time and what is 2 + 2?")
    ]
    
    # Run the agent
    result = agent.invoke({
        "messages": messages,
        "next_step": "agent"
    })
    
    # Print the results
    print("\nConversation:")
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"\nHuman: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\nAI: {message.content}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set up your environment first by running setup.py")
    else:
        run_example() 