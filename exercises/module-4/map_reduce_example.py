from typing import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('../.env')

class MapReduceState(TypedDict):
    documents: List[str]
    summaries: List[str]
    final_summary: str

def create_map_reduce_chain():
    """Create a map-reduce chain for document summarization"""
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    
    def map_function(state: MapReduceState) -> MapReduceState:
        """Map function to summarize individual documents"""
        summaries = []
        for doc in state["documents"]:
            response = chat.invoke([
                SystemMessage(content="Summarize the following text in 2-3 sentences:"),
                HumanMessage(content=doc)
            ])
            summaries.append(response.content)
        return {**state, "summaries": summaries}
    
    def reduce_function(state: MapReduceState) -> MapReduceState:
        """Reduce function to combine summaries"""
        combined_summary = "\n".join(state["summaries"])
        response = chat.invoke([
            SystemMessage(content="Create a coherent summary from these individual summaries:"),
            HumanMessage(content=combined_summary)
        ])
        return {**state, "final_summary": response.content}
    
    # Create workflow
    workflow = StateGraph(MapReduceState)
    
    # Add nodes
    workflow.add_node("map", map_function)
    workflow.add_node("reduce", reduce_function)
    
    # Add edges
    workflow.set_entry_point("map")
    workflow.add_edge("map", "reduce")
    workflow.add_edge("reduce", END)
    
    return workflow.compile()

def run_example():
    """Run an example of map-reduce summarization"""
    # Sample documents
    documents = [
        "The first computer was the ENIAC, completed in 1946. It weighed 30 tons and took up 1,800 square feet.",
        "The first computer mouse was invented by Douglas Engelbart in the 1960s. It was made of wood.",
        "The first email was sent by Ray Tomlinson in 1971. He chose the @ symbol for email addresses."
    ]
    
    # Create and run chain
    chain = create_map_reduce_chain()
    result = chain.invoke({
        "documents": documents,
        "summaries": [],
        "final_summary": ""
    })
    
    # Print results
    print("\nIndividual Summaries:")
    for i, summary in enumerate(result["summaries"], 1):
        print(f"\nDocument {i} Summary:")
        print(summary)
    
    print("\nFinal Combined Summary:")
    print(result["final_summary"])

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set up your environment first by running setup.py")
    else:
        run_example() 