import os
import re
import time
import json
import concurrent.futures
from typing import Dict, Any, List
from datetime import datetime, date

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.state import StateGraph

class SQLAgent:
    """SQL Agent node that generates and executes SQL queries to AWS Redshift in parallel."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SQL Agent with configuration."""
        self.config = config
        # Use boto3 for Redshift data operations
        import boto3
        self.redshift_client = boto3.client('redshift-data', region_name='us-east-1')
        
        # Import LangFuse handler for use in models
        from langfuse.langchain import CallbackHandler
        self.langfuse_handler = CallbackHandler()
        
        # Initialize ChatBedrockConverse for each tool
        self.bedrock_models = {}
        for tool_name, tool_config in self.config["sql_agent"].items():
            self.bedrock_models[tool_name] = ChatBedrockConverse(
                model_id=tool_config["model_id"],
                region_name=tool_config["model_region"],
                max_tokens=tool_config["max_tokens"],
                temperature=tool_config["temperature"],
                top_p=tool_config["top_p"],
                callbacks=[self.langfuse_handler]
            )
        
        # Number of workers for parallel processing
        self.max_workers = 10  # Adjust as needed
    
    def process_sql_queries(self, state: StateGraph) -> StateGraph:
        """Process SQL queries based on classification from supervisor in parallel."""
        state["execution_steps"].append("sql_agent")
        
        # Get classification from the supervisor
        classification = state.get("classification", {})
        tools = classification.get("tools", {})
        
        # Initialize sql_data in state if needed
        if "sql_data" not in state:
            state["sql_data"] = {}
        
        # Check if there are any SQL agent tools to use
        if "sql_agent" not in tools:
            state["execution_steps"].append("sql_agent_no_queries_to_process")
            return state
        
        # Collect all tasks to be executed in parallel
        all_tasks = []
        
        # Process each SQL agent tool and collect queries
        for tool_name, queries in tools["sql_agent"].items():
            if tool_name in self.config["sql_agent"]:
                tool_config = self.config["sql_agent"][tool_name]
                
                # Initialize results array for this tool
                if tool_name not in state["sql_data"]:
                    state["sql_data"][tool_name] = []
                
                # Add tasks for each query
                for query_info in queries:
                    all_tasks.append({
                        "tool_name": tool_name,
                        "tool_config": tool_config,
                        "query": query_info["query"],
                        "ticker": query_info.get("ticker"),
                        "year": query_info.get("year")
                    })
        
        # Execute all queries in parallel
        results = self._process_queries_parallel(all_tasks)
        
        # Organize results back into state structure
        for result in results:
            tool_name = result["tool_name"]
            state["sql_data"][tool_name].append({
                "ticker": result["ticker"],
                "year": result["year"],
                "query": result["query"],
                "result": result["result"]
            })
        
        state["execution_steps"].append("sql_agent_processed")
        return state
    
    def _process_queries_parallel(self, tasks: List[Dict]) -> List[Dict]:
        """Process multiple SQL queries in parallel using ThreadPoolExecutor."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a dictionary to map futures to their original tasks
            future_to_task = {
                executor.submit(
                    self._handle_sql_query,
                    task["query"],
                    task["ticker"],
                    task["year"],
                    task["tool_config"],
                    task["tool_name"]
                ): task for task in tasks
            }
            
            # Process completed futures as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    sql_result = future.result()
                    results.append({
                        "tool_name": task["tool_name"],
                        "query": task["query"],
                        "ticker": task["ticker"],
                        "year": task["year"],
                        "result": sql_result
                    })
                except Exception as e:
                    print(f"Error processing SQL query '{task['query']}': {str(e)}")
                    # Add error results to maintain the structure
                    results.append({
                        "tool_name": task["tool_name"],
                        "query": task["query"],
                        "ticker": task["ticker"],
                        "year": task["year"],
                        "result": {
                            "error": str(e),
                            "status": "ERROR"
                        }
                    })
        
        return results
    
    def _generate_sql_query(self, question: str, tool_config: Dict, tool_name: str) -> str:
        """Generate SQL query using Bedrock models with proper LangChain tracing."""
        try:
            # Load system prompt template from file
            prompt_file = tool_config.get("prompt")
            with open(prompt_file, "r", encoding="utf-8") as file:
                system_prompt = file.read()
            
            model = self.bedrock_models[tool_name]
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]
            
            # Call the model
            response = model.invoke(messages)
            
            # Extract SQL query from response
            query = response.content.strip()
            
            # Clean up SQL query
            query = re.sub(r'^```(?:sql|python|)?\s*', '', query, flags=re.MULTILINE)
            query = re.sub(r'```\s*$', '', query, flags=re.MULTILINE)
            query = query.replace("<sql>", "").replace("</sql>", "")
            query = query.strip()
            
            return query
            
        except Exception as e:
            print(f"Error generating SQL query: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def _execute_sql_query(self, query: str, tool_config: Dict) -> Dict:
        """Execute a SQL query on Redshift and return results."""
        try:
            # Submit query to Redshift
            response = self.redshift_client.execute_statement(
                WorkgroupName=tool_config["workgroup_name"],
                Database=tool_config["database_name"],
                SecretArn=tool_config["secret_arn"],
                Sql=query,
                SessionKeepAliveSeconds=60
            )
            
            query_id = response['Id']
            
            # Wait for query to complete
            max_wait_time = 300  # 5 minutes
            wait_time = 0
            
            while wait_time < max_wait_time:
                status_response = self.redshift_client.describe_statement(Id=query_id)
                status = status_response['Status']
                
                if status == 'FINISHED':
                    break
                elif status == 'FAILED':
                    error_msg = status_response.get('Error', 'Unknown error')
                    raise Exception(f"Query failed: {error_msg}")
                elif status == 'ABORTED':
                    raise Exception("Query was aborted")
                
                # Wait before checking again
                time.sleep(2)
                wait_time += 2
            
            if wait_time >= max_wait_time:
                raise Exception("Query timeout - exceeded maximum wait time")
            
            # Get query results
            result_response = self.redshift_client.get_statement_result(Id=query_id)
            
            # Extract column metadata
            columns = []
            if 'ColumnMetadata' in result_response:
                columns = [col['name'] for col in result_response['ColumnMetadata']]
            
            # Extract rows
            rows = []
            if 'Records' in result_response:
                for record in result_response['Records']:
                    row = []
                    for field in record:
                        # Handle different field types
                        if 'stringValue' in field:
                            row.append(field['stringValue'])
                        elif 'longValue' in field:
                            row.append(field['longValue'])
                        elif 'doubleValue' in field:
                            row.append(field['doubleValue'])
                        elif 'booleanValue' in field:
                            row.append(field['booleanValue'])
                        elif 'isNull' in field and field['isNull']:
                            row.append(None)
                        else:
                            row.append(str(field))
                    rows.append(row)
            
            # Convert to serializable format
            serialized_results = self._serialize_query_results(rows, columns)
            
            return {
                'columns': columns,
                'data': serialized_results,
                'row_count': len(rows),
                'query_id': query_id,
                'status': 'FINISHED'
            }
            
        except Exception as e:
            print(f"Error executing SQL query: {str(e)}")
            return {
                'error': str(e),
                'status': 'ERROR'
            }
    
    def _serialize_query_results(self, results: List, columns: List[str]) -> List[Dict]:
        """Serialize query results to JSON-compatible format with column names."""
        serialized = []
        for row in results:
            row_dict = {}
            for i, value in enumerate(row):
                column_name = columns[i] if i < len(columns) else f"column_{i}"
                if isinstance(value, (datetime, date)):
                    row_dict[column_name] = value.isoformat()
                else:
                    row_dict[column_name] = value
            serialized.append(row_dict)
        return serialized
    
    def _handle_sql_query(self, question: str, ticker: str, year: str, tool_config: Dict, tool_name: str) -> Dict:
        """Generate SQL from the question and execute it."""
        # Enhance the question with any context (ticker, year) if available
        enhanced_question = question
        if ticker:
            enhanced_question += f"\nTicker: {ticker}"
        if year:
            enhanced_question += f"\nYear/Period: {year}" 
        
        # Generate SQL
        sql_query = self._generate_sql_query(enhanced_question, tool_config, tool_name)
        
        # Execute SQL if successfully generated
        if not sql_query.startswith("ERROR:"):
            result = self._execute_sql_query(sql_query, tool_config)
            result["sql_query"] = sql_query  # Include the generated SQL in results
            return result
        else:
            return {
                "error": sql_query,
                "status": "ERROR"
            }