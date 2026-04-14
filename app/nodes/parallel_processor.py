import asyncio
import concurrent.futures
from typing import Dict, Any, List, Callable
from app.core.state import StateGraph

class ParallelProcessor:
    """Node that processes multiple agents in parallel."""
    
    def __init__(self, kb_agent=None, sql_agent=None):
        """Initialize the parallel processor with agent instances."""
        self.kb_agent = kb_agent
        self.sql_agent = sql_agent
        self.max_workers = 10
    
    def process_parallel(self, state: StateGraph) -> StateGraph:
        """Process multiple agents in parallel based on the route."""
        # Record start of parallel processing
        state["execution_steps"].append("parallel_processor_started")
        
        # Get the routes to process
        routes = state.get("route", [])
        if not routes or len(routes) < 2:
            state["execution_steps"].append("parallel_processor_skipped_insufficient_routes")
            return state
            
        tasks = []
        agent_map = {
            "kb_agent": self.kb_agent.process_kb_queries if self.kb_agent else None,
            "sql_agent": self.sql_agent.process_sql_queries if self.sql_agent else None,
        }
        
        # Create a deep copy of state for each agent to avoid conflicts
        # We'll only merge back the specific data fields each agent modifies
        import copy
        
        # Execute agents in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_agent = {}
            for route in routes:
                if route in agent_map and agent_map[route]:
                    # Create a copy of state for each agent
                    agent_state = copy.deepcopy(state)
                    future = executor.submit(agent_map[route], agent_state)
                    future_to_agent[future] = route
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_agent):
                agent_type = future_to_agent[future]
                try:
                    agent_state = future.result()
                    
                    # Merge back the specific data field for this agent
                    if agent_type == "kb_agent":
                        state["kb_data"] = agent_state.get("kb_data", {})
                        state["execution_steps"].append("kb_agent_parallel_processed")
                    elif agent_type == "sql_agent":
                        state["sql_data"] = agent_state.get("sql_data", {})
                        state["execution_steps"].append("sql_agent_parallel_processed")
                    
                    # Merge any new execution steps
                    for step in agent_state.get("execution_steps", []):
                        if step not in state["execution_steps"]:
                            state["execution_steps"].append(step)
                            
                except Exception as e:
                    state["execution_steps"].append(f"{agent_type}_parallel_error: {str(e)}")
                    if agent_type == "kb_agent":
                        state["kb_data"]["error"] = str(e)
                    elif agent_type == "sql_agent":
                        state["sql_data"]["error"] = str(e)
        
        state["execution_steps"].append("parallel_processor_completed")
        return state