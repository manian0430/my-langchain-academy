from typing import TypedDict, Annotated, Sequence, List, Dict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv('../.env')

class ResearchState(TypedDict):
    query: str
    research_notes: List[str]
    outline: Dict[str, List[str]]
    draft: str
    final_report: str

@tool()
def web_search(query: str) -> str:
    """Simulate web search (in real implementation, use actual search API)"""
    # This is a mock implementation
    return f"Found relevant information about {query} from reliable sources..."

@tool()
def save_notes(note: str) -> str:
    """Save research notes"""
    return "Note saved successfully"

def create_research_assistant():
    """Create a research assistant chain"""
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    
    def research_phase(state: ResearchState) -> ResearchState:
        """Conduct initial research"""
        search_result = web_search(state["query"])
        response = chat.invoke([
            SystemMessage(content="You are a research assistant. Analyze this search result and extract key points:"),
            HumanMessage(content=search_result)
        ])
        return {**state, "research_notes": [response.content]}
    
    def outline_phase(state: ResearchState) -> ResearchState:
        """Create an outline based on research"""
        response = chat.invoke([
            SystemMessage(content="Create an outline based on these research notes:"),
            HumanMessage(content="\n".join(state["research_notes"]))
        ])
        # Parse the outline (simplified)
        outline = {"sections": response.content.split("\n")}
        return {**state, "outline": outline}
    
    def draft_phase(state: ResearchState) -> ResearchState:
        """Create initial draft"""
        response = chat.invoke([
            SystemMessage(content="Write a draft based on this outline:"),
            HumanMessage(content=json.dumps(state["outline"]))
        ])
        return {**state, "draft": response.content}
    
    def finalize_phase(state: ResearchState) -> ResearchState:
        """Polish the draft into final report"""
        response = chat.invoke([
            SystemMessage(content="Polish this draft into a final report:"),
            HumanMessage(content=state["draft"])
        ])
        return {**state, "final_report": response.content}
    
    # Create workflow
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("research", research_phase)
    workflow.add_node("outline", outline_phase)
    workflow.add_node("draft", draft_phase)
    workflow.add_node("finalize", finalize_phase)
    
    # Add edges
    workflow.set_entry_point("research")
    workflow.add_edge("research", "outline")
    workflow.add_edge("outline", "draft")
    workflow.add_edge("draft", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

def run_example():
    """Run an example research task"""
    # Create assistant
    assistant = create_research_assistant()
    
    # Initial state
    state = {
        "query": "Impact of artificial intelligence on healthcare",
        "research_notes": [],
        "outline": {},
        "draft": "",
        "final_report": ""
    }
    
    # Run the research process
    result = assistant.invoke(state)
    
    # Print results
    print("\nResearch Notes:")
    print(result["research_notes"][0])
    
    print("\nOutline:")
    print(json.dumps(result["outline"], indent=2))
    
    print("\nDraft:")
    print(result["draft"])
    
    print("\nFinal Report:")
    print(result["final_report"])

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set up your environment first by running setup.py")
    else:
        run_example() 