# LangChain Academy Exercises

This repository contains practical exercises for learning LangChain, organized into modules from basic to advanced concepts.

## Setup

1. Create and activate virtual environment:
```bash
python -m venv lc-academy-env
.\lc-academy-env\Scripts\activate  # Windows
source lc-academy-env/bin/activate  # Unix/MacOS
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Set up environment:
```bash
python module-0/setup.py
```

## Module Overview

### Module 0 - Setup and Basics
**Files:**
- `setup.py`: Environment configuration
- `basics.py`: Basic LangChain concepts
**Features:**
- OpenAI API key setup from .env
- Chat model demonstrations (GPT-3.5, GPT-4)
- Temperature settings (0 for precise, 0.7 for creative)
- Prompt templates
- Basic message handling

### Module 1 - Core Components
**Files:**
- `chain_example.py`: Basic chain implementation
- `agent_example.py`: Agent with tools
**Features:**
- Conversation chains
- Basic tools (get_current_time, calculate)
- StateGraph for workflow
- Agent decision-making
- Message flow handling

### Module 2 - Advanced State & Memory
**Files:**
- `chatbot_memory.py`: Memory-enabled chatbot
**Features:**
- Conversation memory
- User profile management
- State persistence
- Context-aware responses
- State management

### Module 3 - Advanced Features
**Files:**
- `advanced_features.py`: Advanced chatbot features
**Features:**
- Streaming responses
- Breakpoints for debugging
- Timeout handling
- Interruption management
- Async processing

### Module 4 - Advanced Processing
**Files:**
- `map_reduce_example.py`: Parallel processing
- `research_assistant.py`: Research automation
**Features:**
- Map-reduce pattern
- Parallel document summarization
- Research task breakdown
- Multi-step research process
- Result compilation

### Module 5 - Memory Systems
**Files:**
- `memory_agent.py`: In-memory system
- `memory_store.py`: Persistent storage
**Features:**
- In-memory conversation storage
- SQLite-based persistent memory
- User profile management
- Conversation history
- Context-aware responses
- Database integration

### Module 6 - Advanced Assistant Features
**Files:**
- `assistant.py`: Full-featured assistant
**Features:**
- Task analysis
- Async request processing
- Timeout handling
- Multi-step execution
- Result summarization
- Task tracking
- Error handling

## Key Features Across All Modules

1. Environment Management:
   - .env configuration
   - API key handling
   - Environment validation

2. Error Handling:
   - Graceful error management
   - User feedback
   - Timeout handling
   - State recovery

3. Type Safety:
   - TypedDict implementations
   - Type hints
   - State validation

4. Best Practices:
   - Modular design
   - Clean code structure
   - Documentation
   - Example usage

## Running Examples

Each module can be run independently:

```bash
# Run any module example
python module-X/example.py

# For example:
python module-0/basics.py
python module-4/map_reduce_example.py
```

## Learning Path

1. Start with Module 0 to set up your environment and understand basics
2. Progress through modules sequentially
3. Each module builds on concepts from previous ones
4. Complete example exercises in each module
5. Experiment with modifying examples to deepen understanding

## Requirements

See `requirements.txt` for full list of dependencies. Key requirements:
- langchain-openai>=0.0.5
- langchain-core>=0.1.15
- langgraph>=0.0.20
- python-dotenv>=1.0.0

## Notes

- Make sure to have a valid OpenAI API key in your .env file
- Each module includes example usage and documentation
- Feel free to modify examples to experiment with different features
- Check error messages and logs for troubleshooting 