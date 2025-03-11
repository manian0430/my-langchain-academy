from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('../.env')

# Define our state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]

def create_chat_chain():
    """Create a simple chain that processes messages through a chat model"""
    # Initialize our chat model
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Define our agent function
    def agent_function(state: AgentState) -> AgentState:
        """Process the messages and generate a response"""
        # Get response from the chat model
        response = chat.invoke(state["messages"])
        # Add the response to the messages
        return {"messages": [*state["messages"], response]}
    
    # Create our graph
    workflow = StateGraph(AgentState)
    
    # Add our node
    workflow.add_node("agent", agent_function)
    
    # Add our edge
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    
    # Compile the graph
    chain = workflow.compile()
    
    return chain

def run_example():
    """Run an example of our chat chain"""
    # Create the chain
    chain = create_chat_chain()
    
    # Create initial state
    messages = [
        HumanMessage(content="What is the capital of France?")
    ]
    
    # Run the chain
    result = chain.invoke({"messages": messages})
    
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