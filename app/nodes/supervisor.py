import json
import time
import datetime
import pytz
from typing import Dict, Any
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.core.state import StateGraph


class SupervisorNode:
    """Supervisor node that processes user queries using AWS Bedrock."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Define the allowed parameters for the model
        model_params = {
            "temperature": config["supervisor"]["temperature"],
            "top_p": config["supervisor"]["top_p"],
            "max_tokens": config["supervisor"]["max_tokens"]
        }
        
        self.model = ChatBedrockConverse(
            model_id=config["supervisor"]["model_id"],
            region_name=config["supervisor"]["region"],
            **model_params
        )
    
    def process_query(self, state: StateGraph) -> StateGraph:
        """Process the user query and generate a response."""
        try:
            # Record the start of supervisor processing
            ttft_start = time.time()
            
            # Prepare messages for the conversation
            messages = []
            
            # Add system message if available
            system_prompt = self.config["supervisor"]["system_prompt"]
            if system_prompt and system_prompt.strip():
                # Add current date information to system prompt if configured
                if self.config["supervisor"].get("current_date", False):
                    # Get Vietnam local time
                    vietnam_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
                    current_datetime_vietnam = datetime.datetime.now(vietnam_timezone)
                    current_timestamp = current_datetime_vietnam.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Combine system prompt and date information into a single system message
                    combined_prompt = f"{system_prompt}\n\nCurrent timestamp (Vietnam local time): {current_timestamp}. Please consider this when responding to queries that require temporal context."
                    messages.append(SystemMessage(content=combined_prompt))
                else:
                    messages.append(SystemMessage(content=system_prompt))
            
            # Add conversation history if available
            if state.get("conversation_history"):
                for msg in state["conversation_history"]:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
            
            # Add current user query
            messages.append(HumanMessage(content=state["query"]))
            
            # Invoke the model
            response = self.model.invoke(messages)
            
            # Record time to first token
            ttft = time.time() - ttft_start
            
            try:
                classification = json.loads(response.content)
                state["classification"] = classification
                
                # Set the route based on tools in classification
                tools = classification.get("tools", {})
                if classification.get("category") == "OUT_OF_SCOPE":
                    state["route"] = ["out_of_scope"]
                else:
                    # Determine which agents to use based on tools
                    route = []
                    if "kb_agent" in tools:
                        route.append("kb_agent")
                    if "sql_agent" in tools:
                        route.append("sql_agent")
                    
                    # If no specific tools, default to synthesize directly
                    if not route:
                        route = ["synthesize"]
                        
                    state["route"] = route
                    
            except:
                # If parsing fails, set a default classification
                state["classification"] = {
                    "thinking": "Failed to parse response as JSON",
                    "category": "OUT_OF_SCOPE",
                    "response_format": "text",
                    "tools": {
                        "out_of_scope_agent": {
                            "out-of-scope-reason": "Could not classify query correctly",
                            "response_type": "general"
                        }
                    }
                }
                state["route"] = ["out_of_scope"]
                # Store the raw response text
                state["final_response"] = response.content
            
            # Update state with timing
            state["ttft"] = ttft
            state["execution_steps"].append("supervisor_processed")
            
            return state
            
        except Exception as e:
            # Handle errors gracefully
            state["final_response"] = f"Error processing query: {str(e)}"
            state["execution_steps"].append("supervisor_error")
            
            # Set a default out-of-scope classification for errors
            state["classification"] = {
                "thinking": "Error in processing",
                "category": "OUT_OF_SCOPE",
                "response_format": "text",
                "tools": {
                    "out_of_scope_agent": {
                        "out-of-scope-reason": f"System error: {str(e)}",
                        "response_type": "general"
                    }
                }
            }
            state["route"] = ["out_of_scope"]
            
            return state