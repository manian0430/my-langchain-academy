from typing import TypedDict, Annotated, Sequence, List, Dict, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json
import asyncio
from datetime import datetime

# Load environment variables
load_dotenv('../.env')

class AssistantState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    tasks: List[Dict]
    context: Optional[str]
    status: str

class Assistant:
    def __init__(self):
        self.chat = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        self.tasks = []
        self.create_workflow()
    
    def create_workflow(self):
        """Create the assistant workflow"""
        def analyze_request(state: AssistantState) -> AssistantState:
            """Analyze user request and determine required tasks"""
            current_message = state["messages"][-1].content
            
            # Analyze request with AI
            response = self.chat.invoke([
                SystemMessage(content="""You are a task analyzer. 
                Determine what tasks need to be done to fulfill this request.
                Return tasks in this format: task1|task2|task3"""),
                HumanMessage(content=current_message)
            ])
            
            # Parse tasks
            tasks = [
                {"name": task.strip(), "status": "pending"}
                for task in response.content.split("|")
            ]
            
            return {**state, "tasks": tasks, "status": "analyzed"}
        
        def execute_tasks(state: AssistantState) -> AssistantState:
            """Execute the identified tasks"""
            tasks = state["tasks"]
            results = []
            
            for task in tasks:
                # Simulate task execution with AI
                response = self.chat.invoke([
                    SystemMessage(content=f"""You are executing this task: {task['name']}
                    Provide the result of executing this task."""),
                    HumanMessage(content=state["messages"][-1].content)
                ])
                
                results.append({
                    **task,
                    "status": "completed",
                    "result": response.content
                })
            
            return {**state, "tasks": results, "status": "executed"}
        
        def summarize_results(state: AssistantState) -> AssistantState:
            """Summarize task results into a coherent response"""
            task_results = "\n".join(
                f"Task: {task['name']}\nResult: {task['result']}"
                for task in state["tasks"]
            )
            
            response = self.chat.invoke([
                SystemMessage(content="Summarize these task results into a coherent response:"),
                HumanMessage(content=task_results)
            ])
            
            return {
                **state,
                "messages": [*state["messages"], response],
                "status": "completed"
            }
        
        # Create workflow
        workflow = StateGraph(AssistantState)
        
        # Add nodes
        workflow.add_node("analyze", analyze_request)
        workflow.add_node("execute", execute_tasks)
        workflow.add_node("summarize", summarize_results)
        
        # Add edges
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "execute")
        workflow.add_edge("execute", "summarize")
        workflow.add_edge("summarize", END)
        
        self.chain = workflow.compile()
    
    async def process_request(self, message: str, timeout: int = 30):
        """Process a user request with timeout"""
        try:
            task = asyncio.create_task(
                self.chain.ainvoke({
                    "messages": [HumanMessage(content=message)],
                    "tasks": [],
                    "context": None,
                    "status": "started"
                })
            )
            
            result = await asyncio.wait_for(task, timeout=timeout)
            return result
            
        except asyncio.TimeoutError:
            return {
                "messages": [
                    HumanMessage(content=message),
                    AIMessage(content="Request timed out. Please try again.")
                ],
                "tasks": [],
                "context": None,
                "status": "timeout"
            }

def run_example():
    """Run example interactions with the assistant"""
    assistant = Assistant()
    
    # Example requests
    requests = [
        "Can you help me plan a birthday party?",
        "I need help writing a professional email to my boss."
    ]
    
    # Process each request
    for request in requests:
        print(f"\nProcessing request: {request}")
        result = asyncio.run(assistant.process_request(request))
        
        print("\nTasks:")
        for task in result["tasks"]:
            print(f"\nTask: {task['name']}")
            print(f"Status: {task['status']}")
            if "result" in task:
                print(f"Result: {task['result']}")
        
        print("\nFinal Response:")
        if result["messages"]:
            print(result["messages"][-1].content)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set up your environment first by running setup.py")
    else:
        run_example() 