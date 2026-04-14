import operator
from typing import TypedDict, List, Dict, Annotated, Optional

class StateGraph(TypedDict):
    trace_id: str  # placeholder for tracing
    query: str
    user_id: str  # placeholder for user tracking
    conversation_id: str  # placeholder for conversation ID tracking
    message_id: str  # placeholder for message ID tracking
    classification: Dict
    route: List[str]  # new attribute to determine which agent(s) to use
    sql_data: Dict
    kb_data: Dict
    final_response: str
    execution_steps: Annotated[List[str], operator.add]  # record all steps in the graph
    start_time: float
    ttft: float  # time to first token
    execution_time: float
    conversation_history: Optional[List[Dict]]  # store conversation context