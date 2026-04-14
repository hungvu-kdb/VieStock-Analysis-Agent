import boto3
import time
import concurrent.futures
from typing import Dict, Any, List
from app.core.state import StateGraph


class KnowledgeBaseAgent:
    """Agent that retrieves information from AWS Bedrock Knowledge Bases."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kb_configs = config.get("kb_agent", {})
        self.region = next(iter(self.kb_configs.values()), {}).get("region", "us-east-1")
        self.bedrock_agent_runtime = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name=self.region
        )
        # Number of workers for parallel processing
        self.max_workers = 10  # Adjust as needed
        
    def process_kb_queries(self, state: StateGraph) -> StateGraph:
        """Process knowledge base queries in parallel based on supervisor classification."""
        try:
            # Record start time for this operation
            start_time = time.time()
            
            # Initialize kb_data if not already present
            if "kb_data" not in state or not state["kb_data"]:
                state["kb_data"] = {}
                
            classification = state.get("classification", {})
            tools = classification.get("tools", {})
            kb_tools = tools.get("kb_agent", {})
            
            if not kb_tools:
                state["execution_steps"].append("kb_agent_no_queries_to_process")
                return state
            
            # Collect all queries to process in parallel
            all_tasks = []
            
            # Process each KB tool and its queries
            for tool_name, queries in kb_tools.items():
                if tool_name not in self.kb_configs:
                    continue
                    
                kb_config = self.kb_configs[tool_name]
                kb_id = kb_config.get("id")
                max_results = kb_config.get("number_of_vector_search_query_results", 5)
                
                state["kb_data"][tool_name] = []
                
                # Add tasks for each query
                for query_info in queries:
                    query = query_info.get("query", "")
                    if not query:
                        continue
                    
                    all_tasks.append({
                        "tool_name": tool_name,
                        "kb_id": kb_id,
                        "query": query,
                        "ticker": query_info.get("ticker"),
                        "year": query_info.get("year"),
                        "max_results": max_results
                    })
            
            # Execute all queries in parallel
            results = self._process_queries_parallel(all_tasks)
            
            # Organize results back into state structure
            for result in results:
                tool_name = result["tool_name"]
                state["kb_data"][tool_name].append({
                    "query": result["query"],
                    "ticker": result["ticker"],
                    "year": result["year"],
                    "results": result["results"]
                })
            
            # Record the execution step
            state["execution_steps"].append("kb_agent_processed")
            
            return state
            
        except Exception as e:
            state["execution_steps"].append(f"kb_agent_error: {str(e)}")
            state["kb_data"]["error"] = str(e)
            return state
    
    def _process_queries_parallel(self, tasks: List[Dict]) -> List[Dict]:
        """Process multiple KB queries in parallel using ThreadPoolExecutor."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a dictionary to map futures to their original tasks
            future_to_task = {
                executor.submit(
                    self.retrieve_from_knowledge_base,
                    task["kb_id"],
                    task["query"],
                    task["max_results"]
                ): task for task in tasks
            }
            
            # Process completed futures as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    kb_results = future.result()
                    results.append({
                        "tool_name": task["tool_name"],
                        "query": task["query"],
                        "ticker": task["ticker"],
                        "year": task["year"],
                        "results": kb_results
                    })
                except Exception as e:
                    print(f"Error processing query '{task['query']}': {str(e)}")
                    # Add empty results to maintain the structure
                    results.append({
                        "tool_name": task["tool_name"],
                        "query": task["query"],
                        "ticker": task["ticker"],
                        "year": task["year"],
                        "results": []
                    })
        
        return results
    
    def retrieve_from_knowledge_base(self, kb_id: str, query: str, max_results: int = 5) -> List[Dict]:
        """Retrieve information from a specific Knowledge Base."""
        try:
            response = self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=kb_id,
                retrievalQuery={
                    'text': query
                },
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': max_results
                    }
                }
            )
            
            # Process and format the results
            results = []
            for result in response.get('retrievalResults', []):
                content = result.get('content', {})
                document_metadata = {}
                if 'document' in content:
                    document = content['document']
                    document_metadata = {
                        'source': document.get('location', {}).get('s3Location', {}).get('uri', ''),
                        'title': document.get('metadata', {}).get('title', ''),
                        'author': document.get('metadata', {}).get('author', ''),
                        'creation_date': document.get('metadata', {}).get('createdAt', '')
                    }
                
                results.append({
                    'text': content.get('text', ''),
                    'metadata': document_metadata,
                    'score': result.get('score', 0)
                })
            
            return results
        except Exception as e:
            print(f"Error retrieving from knowledge base {kb_id}: {str(e)}")
            return []