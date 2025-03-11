from typing import TypedDict, Annotated, Sequence, List, Dict, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json
import sqlite3
from datetime import datetime

# Load environment variables
load_dotenv('../.env')

class SQLiteMemoryStore:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create messages table
        c.execute('''
        CREATE TABLE IF NOT EXISTS messages
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         user_id TEXT,
         role TEXT,
         content TEXT,
         timestamp TEXT)
        ''')
        
        # Create profiles table
        c.execute('''
        CREATE TABLE IF NOT EXISTS profiles
        (user_id TEXT PRIMARY KEY,
         profile_data TEXT,
         last_updated TEXT)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_message(self, user_id: str, role: str, content: str):
        """Add a message to the database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
        INSERT INTO messages (user_id, role, content, timestamp)
        VALUES (?, ?, ?, ?)
        ''', (user_id, role, content, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_recent_messages(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get recent messages for a user"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
        SELECT role, content, timestamp
        FROM messages
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (user_id, limit))
        
        messages = [{
            "role": role,
            "content": content,
            "timestamp": timestamp
        } for role, content, timestamp in c.fetchall()]
        
        conn.close()
        return list(reversed(messages))
    
    def update_profile(self, user_id: str, profile_data: Dict):
        """Update user profile"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
        INSERT OR REPLACE INTO profiles (user_id, profile_data, last_updated)
        VALUES (?, ?, ?)
        ''', (user_id, json.dumps(profile_data), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_profile(self, user_id: str) -> Dict:
        """Get user profile"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
        SELECT profile_data
        FROM profiles
        WHERE user_id = ?
        ''', (user_id,))
        
        result = c.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return {}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    user_id: str
    memory_store: SQLiteMemoryStore
    context: Optional[str]

def create_persistent_memory_agent():
    """Create an agent with persistent memory storage"""
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    memory_store = SQLiteMemoryStore()
    
    def process_message(state: AgentState) -> AgentState:
        """Process message with persistent memory context"""
        user_id = state["user_id"]
        current_message = state["messages"][-1].content
        
        # Get user profile and recent conversation
        profile = memory_store.get_profile(user_id)
        recent_messages = memory_store.get_recent_messages(user_id)
        
        # Create context-aware prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant with persistent memory. 
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
        
        # Update memory store
        memory_store.add_message(user_id, "user", current_message)
        memory_store.add_message(user_id, "assistant", response.content)
        
        # Update user profile based on conversation
        if "my name is" in current_message.lower():
            name = current_message.lower().split("my name is")[-1].strip()
            profile["name"] = name
            memory_store.update_profile(user_id, profile)
        
        return {
            "messages": [*state["messages"], response],
            "user_id": user_id,
            "memory_store": memory_store,
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
    """Run an example conversation with persistent memory"""
    agent = create_persistent_memory_agent()
    user_id = "user123"
    memory_store = SQLiteMemoryStore()
    
    # First interaction
    result1 = agent.invoke({
        "messages": [HumanMessage(content="Hi! My name is Bob.")],
        "user_id": user_id,
        "memory_store": memory_store,
        "context": None
    })
    
    # Second interaction
    result2 = agent.invoke({
        "messages": [HumanMessage(content="What's my name?")],
        "user_id": user_id,
        "memory_store": result1["memory_store"],
        "context": result1["context"]
    })
    
    # Print results
    print("\nConversation with Persistent Memory:")
    for result in [result1, result2]:
        for message in result["messages"]:
            if isinstance(message, HumanMessage):
                print(f"\nHuman: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"\nAI: {message.content}")
        print(f"\nContext: {result['context']}")
    
    # Show stored messages
    print("\nStored Messages in Database:")
    stored_messages = memory_store.get_recent_messages(user_id)
    for msg in stored_messages:
        print(f"\n{msg['role'].title()}: {msg['content']}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set up your environment first by running setup.py")
    else:
        run_example() 