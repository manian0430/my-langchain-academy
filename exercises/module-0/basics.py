from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('../.env')

def demonstrate_chat_models():
    """Demonstrate different chat models and temperature settings"""
    # Initialize chat models with different settings
    gpt4_chat = ChatOpenAI(model="gpt-4", temperature=0)  # More precise
    gpt35_chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)  # More creative
    
    # Example messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is LangChain?")
    ]
    
    print("GPT-4 Response (temperature=0):")
    print(gpt4_chat.invoke(messages))
    print("\nGPT-3.5 Response (temperature=0.7):")
    print(gpt35_chat.invoke(messages))

def demonstrate_prompt_templates():
    """Demonstrate working with prompt templates"""
    # Create a chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a {role} who speaks like {style}."),
        ("human", "{question}")
    ])
    
    # Format the prompt with variables
    formatted_messages = prompt.format_messages(
        role="teacher",
        style="Shakespeare",
        question="What is artificial intelligence?"
    )
    
    # Use the formatted prompt with a chat model
    chat = ChatOpenAI()
    response = chat.invoke(formatted_messages)
    print("\nPrompt Template Response:")
    print(response)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set up your environment first by running setup.py")
    else:
        print("=== Chat Models Demo ===")
        demonstrate_chat_models()
        print("\n=== Prompt Templates Demo ===")
        demonstrate_prompt_templates() 