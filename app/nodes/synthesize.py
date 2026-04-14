import time
from typing import Dict, Any
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.state import StateGraph

class SynthesizeNode:
    """Node to synthesize results from KB agent and SQL agent into a coherent response."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        model_params = {
            "temperature": config["synthesize"]["temperature"],
            "top_p": config["synthesize"]["top_p"],
            "max_tokens": config["synthesize"]["max_tokens"]
        }
        
        self.model = ChatBedrockConverse(
            model_id=config["synthesize"]["model_id"],
            region_name=config["synthesize"]["region"],
            **model_params
        )
    
    def synthesize_results(self, state: StateGraph) -> StateGraph:
        """Synthesize results from KB agent and SQL agent into a coherent response."""
        try:
            # Record start time for synthesis
            start_time = time.time()
            
            # Extract data from KB agent and SQL agent
            kb_data = state.get("kb_data", {})
            sql_data = state.get("sql_data", {})
            
            # Prepare system prompt
            system_prompt = self.config["synthesize"]["system_prompt"]
            
            # Prepare context for the model
            context = {
                "query": state["query"],
                "kb_data": kb_data,
                "sql_data": sql_data
            }
            
            # Format context as string
            context_str = f"""
User Query: {context['query']}

Knowledge Base Results:
{context['kb_data']}

SQL Query Results:
{context['sql_data']}
"""
            
            # Invoke the model
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Please synthesize the following information into a coherent response:\n\n{context_str}")
            ]
            
            response = self.model.invoke(messages)
            
            # Update the state with the synthesized response
            state["final_response"] = response.content
            state["execution_steps"].append("synthesize_processed")
            
            # Record synthesis duration
            synthesis_time = time.time() - start_time
            state["execution_steps"].append(f"synthesis_time: {synthesis_time:.2f}s")
            
            return state
            
        except Exception as e:
            # Handle errors gracefully
            state["final_response"] = f"Error synthesizing results: {str(e)}"
            state["execution_steps"].append("synthesize_error")
            return state