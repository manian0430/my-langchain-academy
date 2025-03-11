from typing import TypedDict, Annotated, Sequence, List, Dict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv('../.env')

class ChatMemory:
    def __init__(self, file_path: str = "chat_history.json"):
        self.file_path = file_path
        self.load_history()
    
    def load_history(self):
        """Load chat history from file"""
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {}
    
    def save_history(self):
        """Save chat history to file"""
        with open(self.file_path, 'w') as f:
            json.dump(self.history, f)
    
    def add_conversation(self, user_id: str, messages: List[Dict]):
        """Add a conversation to history"""
        if user_id not in self.history:
            self.history[user_id] = []
        self.history[user_id].extend(messages)
        self.save_history()
    
    def get_conversation(self, user_id: str) -> List[Dict]:
        """Get conversation history for a user"""
        return self.history.get(user_id, [])

class ChatbotState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    user_id: str
    memory: ChatMemory

def create_chatbot():
    """Create a chatbot with memory"""
    # Initialize our chat model and memory
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    memory = ChatMemory()
    
    def process_message(state: ChatbotState) -> ChatbotState:
        """Process message and update memory"""
        # Get conversation history
        history = memory.get_conversation(state["user_id"])
        
        # Create prompt with history context
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Previous conversation context: {history}"),
            ("human", "{input}")
        ])
        
        # Format messages
        messages = prompt.format_messages(
            history=str(history),
            input=state["messages"][-1].content
        )
        
        # Get response
        response = chat.invoke(messages)
        
        # Update memory
        new_messages = [
            {"role": "user", "content": state["messages"][-1].content},
            {"role": "assistant", "content": response.content}
        ]
        memory.add_conversation(state["user_id"], new_messages)
        
        return {
            "messages": [*state["messages"], response],
            "user_id": state["user_id"],
            "memory": memory
        }
    
    # Create our graph
    workflow = StateGraph(ChatbotState)
    
    # Add our node
    workflow.add_node("process", process_message)
    
    # Add our edges
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    
    # Compile the graph
    chain = workflow.compile()
    
    return chain

def run_example():
    """Run an example conversation"""
    chatbot = create_chatbot()
    user_id = "user123"
    
    # First message
    result1 = chatbot.invoke({
        "messages": [HumanMessage(content="Hi! My name is Alice.")],
        "user_id": user_id,
        "memory": ChatMemory()
    })
    
    # Second message
    result2 = chatbot.invoke({
        "messages": [HumanMessage(content="What's my name?")],
        "user_id": user_id,
        "memory": result1["memory"]
    })
    
    # Print results
    print("\nConversation:")
    for message in result1["messages"] + result2["messages"]:
        if isinstance(message, HumanMessage):
            print(f"\nHuman: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\nAI: {message.content}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set up your environment first by running setup.py")
    else:
        run_example() 