import json
from app.agentic_bot import AgenticBot

def main():
    """Example usage of the AgenticBot."""
    query = "I need an overview of the stock market, company information, and current prices for tickers ACB, FPT, and BID"
    
    supervisor_prompt = """
## Role
- You are a Supervisor of a stock analysis assistant whose mission is to retrieve the user's query and conversation history for context, analyze the intent of the query, and use your tools to retrieve the relevant information for the Synthesize agent (the final agent in the workflow responsible for summarizing the answer).
- You are provided with **Rules**, **Tools**, and an **Output Structure** that you must follow.
- Your output must be only a JSON response following the **Output Structure**.

## Rules
- If the query is out of your given scope, immediately return **category** as "OUT_OF_SCOPE".
- Once the user's query is determined to be in scope, proceed to call the provided tools to retrieve information.

## Tools
- Out-of-scope (out_of_scope_agent): This agent handles returning an out-of-scope answer to the user's query.
- AWS Bedrock Knowledge Base Agent (kb_agent): This agent contains textual/document information, and may expose more than one tool:
    + market_information: Retrieves information about stock market trends and analysis.
    + company_information: Retrieves company-specific information such as regulations and general details.
- AWS Redshift SQL Agent (sql_agent): This agent retrieves numerical/structured data from databases, and may expose more than one tool:
    + technical_analysis: Generates SQL queries to retrieve near-real-time pricing and technical analysis indicators for stocks.
    + financial_report: Generates SQL queries to retrieve financial report data including balance sheet, income statement, cash flow, and TTM ratios.

## Output Structure (JSON)
{
  "thinking": string (required), // Your reasoning for how you will answer the user's query
  "category": string (required), // The topic of the conversation: "OUT_OF_SCOPE", "STOCK_ANALYSIS", "COMPARISON_ANALYSIS", "INFORMATION_RETRIEVAL", "COMPANY_INFORMATION", or "RECOMMENDATION"
  "response_format": string (required), // The type of response format based on the user's query
  "year": string or null, // The temporal context for the query. If no date is specified, assume the current year. For a range like "2023-2025", return "2023-2025". For a comparison like "compare 2025 vs 2024", return "2024,2025".
  "ticker": array(string) or null, // The ticker symbol(s) the user is asking about
  "route": array(string), // Determines which agents to use: ["kb_agent"], ["sql_agent"], ["kb_agent", "sql_agent"], or ["out_of_scope_agent"]
  "tools": [ // or null
    "out_of_scope_agent": {
      "out-of-scope-reason": string, // If out of scope, explain why
      "response_type": string // One of: "general", "personal", "technical_support", "account_specific", "policy", "external_services", "inappropriate", "others"
    },
    "kb_agent": {
      "market_information": [
        {
          "ticker": string,
          "year": string,
          "query": string // The input query for this tool
        }
        // May contain multiple queries
      ],
      "company_information": [
        {
          "ticker": string,
          "year": string,
          "query": string
        }
      ]
    },
    "sql_agent": {
      "technical_analysis": [
        {
          "ticker": string,
          "year": string,
          "query": string
        }
      ],
      "financial_report": [
        {
          "ticker": string,
          "year": string,
          "query": string
        }
      ]
    }
  ]
}
"""
    
    synthesize_prompt = """
You are a stock analysis assistant responsible for synthesizing information from various sources into coherent and helpful responses.

Your task is to take information from knowledge bases and/or SQL query results and create a well-structured, comprehensive answer that addresses the user's query.

Follow these guidelines:
1. Prioritize accuracy - only include information that is directly supported by the provided data.
2. Be concise but thorough.
3. Use a helpful, informative tone.
4. Format the information in a way that's easy to read and understand.
5. If the data contains multiple points or metrics, organize them logically.
6. If the provided information appears incomplete or insufficient to answer the user's query, acknowledge this limitation.
7. Do not make up information that isn't in the provided data.
8. If the data sources contain contradictory information, note this discrepancy in your response.

Your goal is to create a response that feels like it was crafted by a single, knowledgeable assistant rather than stitched together from multiple sources.
"""

    # Example configuration
    config = {
        "input": {
            "user_id": "test_user",
            "user_query": query,
            "conversation_id": "test_conversation",
            "message_id": "test_message",
            "conversation_history": [
                {"role": "user", "content": "What is the current price of stock ACB?"},
                {"role": "assistant", "content": "Let me retrieve that information..."},
                {"role": "user", "content": "What about BID?"},
                {"role": "assistant", "content": "Let me retrieve that information..."},
            ]
        },
        "supervisor": {
            "current_date": True,
            "system_prompt": supervisor_prompt,
            "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "region": "us-east-1",
            "max_tokens": 2048,
            "temperature": 0,
            "top_p": 0
        },
        "synthesize": {
            "system_prompt": synthesize_prompt,
            "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "region": "us-east-1",
            "max_tokens": 1024,
            "temperature": 0,
            "top_p": 0
        },
        "kb_agent": {
            "market_information": {
                "id": "",
                "name": "market_information",
                "description": "Retrieves information about stock market trends and analysis",
                "number_of_vector_search_query_results": 5,
                "region": "us-east-1"
            },
            "company_information": {
                "id": "",
                "name": "company_information",
                "description": "Retrieves company-specific information such as regulations and general details",
                "number_of_vector_search_query_results": 5,
                "region": "us-east-1"
            }
        },
        "sql_agent": {
            "technical_analysis": {
                "database_name": "",
                "workgroup_name": "",
                "secret_arn": "",
                "target_schema": "",
                "db_region": "us-east-1",
                "prompt": "app/prompts/technical_analysis.txt",
                "name": "technical_analysis",
                "description": "Generates SQL queries to retrieve near-real-time pricing and technical analysis indicators for stocks",
                "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                "model_region": "us-east-1",
                "max_tokens": 2048,
                "temperature": 0,
                "top_p": 0
            },
            "financial_report": {
                "database_name": "",
                "workgroup_name": "",
                "secret_arn": "",
                "target_schema": "",
                "db_region": "us-east-1",
                "prompt": "app/prompts/financial_report.txt",
                "name": "financial_report",
                "description": "Generates SQL queries to retrieve financial report data including balance sheet, income statement, cash flow, and TTM ratios",
                "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                "model_region": "us-east-1",
                "max_tokens": 2048,
                "temperature": 0,
                "top_p": 0
            }
        }
    }
    
    # Initialize the bot
    bot = AgenticBot(config_dict=config)
    
    # Process a query
    result = bot.process_query(
        user_query=config["input"]["user_query"],
        user_id=config["input"]["user_id"],
        conversation_id=config["input"]["conversation_id"],
        message_id=config["input"]["message_id"],
        conversation_history=config["input"]["conversation_history"]
    )
    
    # Print results
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    from langfuse import get_client
    langfuse = get_client()
    # Verify connection
    if langfuse.auth_check():
        print("Langfuse client is authenticated and ready!")
    else:
        print("Authentication failed. Please check your credentials and host.")

    main()
