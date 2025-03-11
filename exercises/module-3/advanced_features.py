from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv
import asyncio
import time
import os

# Load environment variables
load_dotenv('../.env')

class StreamingHandler(BaseCallbackHandler):
    """Handler for streaming responses"""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Print tokens as they're generated"""
        print(token, end="", flush=True)

class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    breakpoint: Optional[str]
    interrupted: bool

class AdvancedChatbot:
    def __init__(self):
        self.chat = ChatOpenAI(
            model="gpt-3.5-turbo",
            streaming=True,
            callbacks=[StreamingHandler()]
        )
        self.create_workflow()
    
    def create_workflow(self):
        """Create the chatbot workflow"""
        def process_message(state: ChatState) -> ChatState:
            """Process message with streaming and breakpoint support"""
            # Check for breakpoint
            if state.get("breakpoint"):
                print(f"\n[BREAKPOINT] {state['breakpoint']}")
                return state
            
            # Check for interruption
            if state.get("interrupted"):
                print("\n[INTERRUPTED]")
                return state
            
            # Process message
            response = self.chat.invoke(state["messages"])
            return {
                "messages": [*state["messages"], response],
                "breakpoint": None,
                "interrupted": False
            }
        
        def check_breakpoint(state: ChatState) -> str:
            """Check if we should continue or hit a breakpoint"""
            if state.get("breakpoint"):
                return "breakpoint_handler"
            return "process"
        
        def handle_breakpoint(state: ChatState) -> ChatState:
            """Handle breakpoint"""
            print(f"\nHandling breakpoint: {state['breakpoint']}")
            # Clear breakpoint and continue
            return {
                **state,
                "breakpoint": None
            }
        
        # Create graph
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("process", process_message)
        workflow.add_node("breakpoint_handler", handle_breakpoint)
        
        # Add edges
        workflow.set_entry_point("process")
        workflow.add_conditional_edges(
            "process",
            check_breakpoint
        )
        workflow.add_edge("breakpoint_handler", "process")
        
        # Compile
        self.chain = workflow.compile()
    
    async def chat_with_timeout(self, message: str, timeout: int = 5):
        """Chat with timeout support"""
        try:
            # Create task for chat
            task = asyncio.create_task(
                self.chain.ainvoke({
                    "messages": [HumanMessage(content=message)],
                    "breakpoint": None,
                    "interrupted": False
                })
            )
            
            # Wait for response with timeout
            await asyncio.wait_for(task, timeout=timeout)
            
        except asyncio.TimeoutError:
            print("\n[TIMEOUT] Response took too long")
            return {
                "messages": [
                    HumanMessage(content=message),
                    AIMessage(content="[Response interrupted due to timeout]")
                ],
                "breakpoint": None,
                "interrupted": True
            }

def run_example():
    """Run examples of advanced features"""
    chatbot = AdvancedChatbot()
    
    # Example 1: Normal conversation
    print("\nExample 1: Normal conversation")
    result = chatbot.chain.invoke({
        "messages": [HumanMessage(content="What is LangChain?")],
        "breakpoint": None,
        "interrupted": False
    })
    
    # Example 2: Conversation with breakpoint
    print("\nExample 2: Conversation with breakpoint")
    result = chatbot.chain.invoke({
        "messages": [HumanMessage(content="Tell me about AI")],
        "breakpoint": "Check AI definition",
        "interrupted": False
    })
    
    # Example 3: Conversation with timeout
    print("\nExample 3: Conversation with timeout")
    asyncio.run(chatbot.chat_with_timeout(
        "Write a very long essay about the history of AI",
        timeout=3
    ))

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set up your environment first by running setup.py")
    else:
        run_example() 