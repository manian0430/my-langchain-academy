from typing import TypedDict, Annotated, Sequence, List, Dict, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# Load environment variables
load_dotenv('../.env')

class Memory:
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        self.user_profiles: Dict[str, Dict] = {}
        self.last_interaction: Dict[str, datetime] = {}
    
    def add_message(self, user_id: str, message: Dict):
        """Add a message to user's conversation history"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        self.conversations[user_id].append(message)
        self.last_interaction[user_id] = datetime.now()
    
    def get_recent_messages(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get recent messages for a user"""
        if user_id not in self.conversations:
            return []
        return self.conversations[user_id][-limit:]
    
    def update_user_profile(self, user_id: str, info: Dict):
        """Update user profile information"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}
        self.user_profiles[user_id].update(info)
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Get user profile information"""
        return self.user_profiles.get(user_id, {})

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    user_id: str
    memory: Memory
    context: Optional[str]

def create_memory_agent():
    """Create an agent with memory capabilities"""
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    memory = Memory()
    
    def process_message(state: AgentState) -> AgentState:
        """Process message with memory context"""
        user_id = state["user_id"]
        current_message = state["messages"][-1].content
        
        # Get user profile and recent conversation
        profile = memory.get_user_profile(user_id)
        recent_messages = memory.get_recent_messages(user_id)
        
        # Create context-aware prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant with memory. 
            User Profile: {profile}
            Recent Conversation: {conversation}
            Respond naturally and use the context when appropriate."""),
            ("human", "{input}")
        ])
        
        # Format messages with context
        messages = prompt.format_messages(
            profile=json.dumps(profile),
            conversation=json.dumps(recent_messages),
            input=current_message
        )
        
        # Get response
        response = chat.invoke(messages)
        
        # Update memory
        memory.add_message(user_id, {
            "role": "user",
            "content": current_message,
            "timestamp": datetime.now().isoformat()
        })
        memory.add_message(user_id, {
            "role": "assistant",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update user profile based on conversation
        # This is a simple example - in real applications, you'd want more sophisticated profile updates
        if "my name is" in current_message.lower():
            name = current_message.lower().split("my name is")[-1].strip()
            memory.update_user_profile(user_id, {"name": name})
        
        return {
            "messages": [*state["messages"], response],
            "user_id": user_id,
            "memory": memory,
            "context": f"Profile: {json.dumps(profile)}"
        }
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add node
    workflow.add_node("process", process_message)
    
    # Add edges
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    
    return workflow.compile()

def run_example():
    """Run an example conversation with memory"""
    agent = create_memory_agent()
    user_id = "user123"
    
    # First interaction
    result1 = agent.invoke({
        "messages": [HumanMessage(content="Hi! My name is Alice.")],
        "user_id": user_id,
        "memory": Memory(),
        "context": None
    })
    
    # Second interaction
    result2 = agent.invoke({
        "messages": [HumanMessage(content="What's my name?")],
        "user_id": user_id,
        "memory": result1["memory"],
        "context": result1["context"]
    })
    
    # Print results
    print("\nConversation with Memory:")
    for result in [result1, result2]:
        for message in result["messages"]:
            if isinstance(message, HumanMessage):
                print(f"\nHuman: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"\nAI: {message.content}")
        print(f"\nContext: {result['context']}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set up your environment first by running setup.py")
    else:
        run_example() 