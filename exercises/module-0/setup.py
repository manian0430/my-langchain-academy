import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def setup_environment():
    """Setup the environment with necessary API keys from .env file"""
    # Load environment variables from .env file
    load_dotenv('../.env')  # Load from exercises/.env
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in .env file")
        return False
    
    # Test the environment
    try:
        chat = ChatOpenAI(model="gpt-3.5-turbo")
        print("✅ Environment setup successful!")
        print("✅ OpenAI API key loaded from .env file")
        return True
    except Exception as e:
        print(f"❌ Environment setup failed: {str(e)}")
        return False

if __name__ == "__main__":
    setup_environment() 