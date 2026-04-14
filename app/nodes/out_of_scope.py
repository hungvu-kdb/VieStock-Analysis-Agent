import time
from typing import Dict, Any
from app.core.state import StateGraph


class OutOfScopeNode:
    """Node that handles out-of-scope queries without LLM involvement."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Define predefined responses for common out-of-scope scenarios
        self.out_of_scope_responses = {
            "general": "I'm sorry, I can only assist with information related to the stock market and financial analysis. Please ask something within that scope.",
            "personal": "I'm sorry, I cannot answer personal questions. If you have any questions about the stock market or financial analysis, I'd be happy to help.",
            "technical_support": "For technical issues, login problems, or account management, please contact technical support directly for the fastest and most accurate assistance.",
            "account_specific": "To protect your personal information, I cannot access or share account-specific details. Please contact customer support or log in to your account to check.",
            "policy": "For the most accurate information about policies and regulations, please refer to the official documentation or contact the relevant department.",
            "external_services": "I'm sorry, I don't have information about external services. Please contact the relevant service provider directly.",
            "inappropriate": "I cannot assist with this request. Please ask a different question that I can help you with.",
            "others": "I cannot respond to your current request. Please ask a question related to the stock market or financial analysis."
        }
    
    def process_out_of_scope(self, state: StateGraph) -> StateGraph:
        """Process out-of-scope queries and return predefined responses."""
        try:
            # Record the start of processing
            process_start = time.time()
            
            # Get the classification from state
            classification = state.get("classification", {})
            
            # Extract information from supervisor's classification
            if classification and classification.get("category") == "OUT_OF_SCOPE" and "tools" in classification:
                out_of_scope_data = classification.get("tools", {}).get("out_of_scope_agent", {})
                response_type = out_of_scope_data.get("response_type", "general")
            else:
                # Default to general if no specific type is provided
                response_type = "general"
            
            # Get the appropriate response
            response = self.out_of_scope_responses.get(
                response_type, 
                self.out_of_scope_responses["general"]
            )
            
            # Update state with response
            state["final_response"] = response
            state["execution_steps"].append("out_of_scope_processed")
            
            # Record processing time (should be minimal since no LLM is involved)
            processing_time = time.time() - process_start
            state["ttft"] = processing_time
            
            return state
            
        except Exception as e:
            # Handle errors gracefully
            state["final_response"] = f"I'm sorry, but I encountered an error processing your request: {str(e)}"
            state["execution_steps"].append("out_of_scope_error")
            return state
