import json
import time
from ulid import ULID
from typing import Dict, Any
from langgraph.graph import StateGraph as LangStateGraph, START, END
from app.core.state import StateGraph
from app.nodes.supervisor import SupervisorNode
from app.nodes.out_of_scope import OutOfScopeNode
from app.nodes.kb_agent import KnowledgeBaseAgent
from app.nodes.sql_agent import SQLAgent
from app.nodes.parallel_processor import ParallelProcessor
from app.nodes.synthesize import SynthesizeNode

# Observation
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()


class AgenticBot:
    """Main class for the Agentic AI Bot using LangGraph."""
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """Initialize the bot with configuration."""
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        self.supervisor = SupervisorNode(self.config)
        self.out_of_scope = OutOfScopeNode(self.config)
        self.kb_agent = KnowledgeBaseAgent(self.config)
        self.sql_agent = SQLAgent(self.config)
        self.synthesize = SynthesizeNode(self.config)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> LangStateGraph:
        """Build the LangGraph workflow."""
        # Create the graph
        workflow = LangStateGraph(StateGraph)
        
        # Initialize the parallel processor
        self.parallel_processor = ParallelProcessor(kb_agent=self.kb_agent, sql_agent=self.sql_agent)
        
        # Add the nodes
        workflow.add_node("supervisor", self.supervisor.process_query)
        workflow.add_node("out_of_scope", self.out_of_scope.process_out_of_scope)
        workflow.add_node("kb_agent", self.kb_agent.process_kb_queries)
        workflow.add_node("sql_agent", self.sql_agent.process_sql_queries)
        workflow.add_node("parallel_processor", self.parallel_processor.process_parallel)
        workflow.add_node("synthesize", self.synthesize.synthesize_results)
        
        # Define the flow: START -> supervisor
        workflow.add_edge(START, "supervisor")
        
        # Conditional routing based on supervisor's route classification
        def route_based_on_classification(state: StateGraph) -> str:
            route = state.get("route", [])
            
            # Handle out-of-scope
            if "out_of_scope" in route:
                return "out_of_scope"
                
            # Handle parallel processing if multiple routes
            if len(route) > 1:
                return "parallel_processor"
                
            # Handle single agent routes
            if route == ["kb_agent"]:
                return "kb_agent"
            elif route == ["sql_agent"]:
                return "sql_agent"
            
            # Default to synthesize
            return "synthesize"
        
        # Add conditional routing from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            route_based_on_classification
        )
        
        # Connect out_of_scope to END
        workflow.add_edge("out_of_scope", END)
        
        # Connect kb_agent to synthesize
        workflow.add_edge("kb_agent", "synthesize")
        
        # Connect sql_agent to synthesize
        workflow.add_edge("sql_agent", "synthesize")
        
        # Connect parallel_processor to synthesize
        workflow.add_edge("parallel_processor", "synthesize")
        
        # Connect synthesize to END
        workflow.add_edge("synthesize", END)
        
        # Compile the graph
        graph = workflow.compile().with_config({"callbacks": [langfuse_handler]})
        return graph
    
    def _prepare_initial_state(self, user_query: str, user_id: str = "test_user", conversation_id: str = "test_conversation", message_id: str = "test_message", conversation_history: list = None) -> StateGraph:
        """Prepare the initial state for graph execution."""
        return StateGraph(
            trace_id=str(ULID()),
            query=user_query,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            classification={},
            route=[],
            sql_data={},
            kb_data={},
            final_response="",
            execution_steps=[],
            start_time=time.time(),
            ttft=0.0,
            execution_time=0.0,
            conversation_history=conversation_history or []
        )
    
    def process_query(self, user_query: str, user_id: str = "test_user", conversation_id: str = "test_conversation", message_id: str = "test_message", conversation_history: list = None) -> Dict[str, Any]:
        """Process a user query through the graph workflow."""
        # Prepare initial state
        initial_state = self._prepare_initial_state(user_query, user_id, conversation_id, message_id, conversation_history)
        
        try:
            # Execute the graph
            result = self.graph.invoke(initial_state)
            
            # Calculate total execution time
            result["execution_time"] = time.time() - result["start_time"]
            
            # If KB data was retrieved, format it as the final response for now
            if result.get("kb_data") and not result.get("final_response"):
                result["final_response"] = f"Retrieved information from knowledge base: {json.dumps(result['kb_data'], indent=2)}"
            
            return {
                "trace_id": result["trace_id"],
                "user_id": result["user_id"],
                "conversation_id": result["conversation_id"],
                "message_id": result["message_id"],
                "query": result["query"],
                "response": result["final_response"],
                "execution_steps": result["execution_steps"],
                "timing": {
                    "ttft": result["ttft"],
                    "execution_time": result["execution_time"]
                },
                "kb_data": result.get("kb_data", {})
            }
            
        except Exception as e:
            return {
                "trace_id": initial_state["trace_id"],
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "query": user_query,
                "response": f"Error processing query: {str(e)}",
                "execution_steps": ["error"],
                "timing": {
                    "ttft": 0.0,
                    "execution_time": time.time() - initial_state["start_time"]
                }
            }