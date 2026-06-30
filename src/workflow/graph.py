from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from src.core.guardrails import validate_ticker
from src.agents.market_analyst import get_market_agent
from src.agents.news_synthesizer import get_news_synthesizer_agent
from src.agents.finance_qa import get_finance_qa_agent
from src.agents.portfolio_analyst import get_portfolio_analyst_agent
from src.agents.goal_planner import get_goal_planner_agent
from src.core.guardrails import check_safety

# Define State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    is_financial_query: bool
    query_type: str 
    remaining_tasks: list[str] 
    market_task_query: str # Decomposed query for market analyst
    news_task_query: str   # Decomposed query for news synthesizer
    finance_qa_task_query: str # Decomposed query for finance Q&A
    portfolio_task_query: str # Decomposed query for portfolio analyst
    goal_task_query: str # Decomposed query for goal planner
    market_report: str # Final summary from market analyst
    news_report: str   # Final summary from news synthesizer
    education_report: str # Final summary from finance educator
    portfolio_report: str # Final summary from portfolio analyst
    goal_report: str # Final summary from goal planner

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from pydantic import BaseModel, Field

class RouteTasks(BaseModel):
    needs_market: bool = Field(description="True if the query needs numbers, price, P/E, or technical data.")
    market_query: str = Field(description="The specific part of the user request for the market specialist (e.g., 'Price and P/E ratio for MSFT'). Empty if not needed.")
    needs_news: bool = Field(description="True if the query needs recent headlines, sentiment, or events.")
    news_query: str = Field(description="The specific part of the user request for the news specialist (e.g., 'Summarize news about AI data centers for MSFT'). Empty if not needed.")
    needs_education: bool = Field(description="True if the query is asking for definitions, concepts, or educational material (e.g., 'What is P/E', 'Explain bonds').")
    education_query: str = Field(description="The specific part of the user request for the finance educator. Empty if not needed.")
    needs_portfolio: bool = Field(description="True if the query asks to review, analyze, or suggest changes to a user's current portfolio composition or asset allocation.")
    portfolio_query: str = Field(description="The specific part of the user request for the portfolio analyst (e.g. 'Analyze my current holdings'). Do NOT include mathematical goal-seeking or retirement calculations here.")
    needs_goal: bool = Field(description="True if the query is about future goals, calculating required savings, ROI needed, or amortization.")
    goal_query: str = Field(description="The specific part of the user request for the goal planner (e.g. 'Calculate how much I need to save monthly').")

# Node: Guardrails
def guardrails_node(state: AgentState):
    """
    Decomposes the user query into specific tasks for each agent.
    """
    messages = state["messages"]
    if not messages:
        return {"is_financial_query": False, "query_type": "unrelated", "remaining_tasks": []}

    # --- SECURITY CHECK ---
    last_user_msg = messages[-1].content
    
    # Run simple safety check (PII + Jailbreak regex)
    safety_result = check_safety(last_user_msg)
    
    if not safety_result["safe"]:
        return {
            "is_financial_query": False, 
            "query_type": "blocked",
            "remaining_tasks": [],
            "messages": [AIMessage(content=safety_result["message"])]
        }
    # ----------------------
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    structured_llm = llm.with_structured_output(RouteTasks)
    
    system_prompt = (
        "You are a routing and decomposition expert for a financial assistant. "
        "Analyze the user's LATEST query and split it into three specific assignments:\n"
        "1. MARKET_QUERY: Strictly about numbers, prices, P/E, calculations, and fundamentals. This includes current status and prices for commodities (Gold, Oil, etc.).\n"
        "2. NEWS_QUERY: Strictly about current events, headlines, 'what happened', and sentiment. For 'What is happening with [Asset]?', you should usually trigger BOTH market and news.\n"
        "3. EDUCATION_QUERY: Questions about definitions, concepts, theory, or 'how things work' (e.g. 'What is a stock?').\n"
        "4. PORTFOLIO_QUERY: Deep analysis of a collection of stocks/assets, risk assessment, and improvement suggestions.\n"
        "5. GOAL_QUERY: Financial planning, targets, ROI calculations, savings paths, and amortization schedules.\n\n"
        "JOINT QUERIES: If a user asks a complex question like 'How much to save for $1M considering my current portfolio?', "
        "set BOTH needs_goal=True and needs_portfolio=True. Split the request accordingly.\n\n"
        "If a user asks 'How much do I need to save to have $1M in 10 years?', set needs_goal=True and put details in GOAL_QUERY.\n"
        "If a user asks 'Review my portfolio of 50% NVDA and 50% Gold', set needs_portfolio=True and put the portfolio details in PORTFOLIO_QUERY.\n"
        "If a user asks 'What is happening with [Asset]?', set BOTH needs_market=True and needs_news=True.\n"
        "Respond with all needed queries. If unrelated to finance, set all needs_* to False."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"messages": messages[-5:]})
    
    tasks = []
    if result.needs_market: tasks.append("market")
    if result.needs_news: tasks.append("news")
    if result.needs_education: tasks.append("education")
    if result.needs_portfolio: tasks.append("portfolio")
    if result.needs_goal: tasks.append("goal")
    
    if tasks:
        return {
            "is_financial_query": True, 
            "query_type": tasks[0],
            "remaining_tasks": tasks,
            "market_task_query": result.market_query,
            "news_task_query": result.news_query,
            "finance_qa_task_query": result.education_query,
            "portfolio_task_query": result.portfolio_query,
            "goal_task_query": result.goal_query,
            # Reset reports to prevent context leak from previous turns
            "market_report": "",
            "news_report": "",
            "education_report": "",
            "portfolio_report": "",
            "goal_report": ""
        }
    else:
        return {
            "is_financial_query": False, 
            "query_type": "unrelated",
            "remaining_tasks": [],
            "market_report": "",
            "news_report": "",
            "education_report": "",
            "portfolio_report": "",
            "goal_report": "",
            "messages": [AIMessage(content="I specialized in financial market analysis, news synthesis, goal planning, and portfolio review. Please ask me about stocks, market trends, savings goals, or your investment portfolio!")]
        }

# Nodes: Agents
market_agent = get_market_agent()
news_agent = get_news_synthesizer_agent()
finance_qa_agent = get_finance_qa_agent()
portfolio_agent = get_portfolio_analyst_agent()
goal_agent = get_goal_planner_agent()

def market_agent_node(state: AgentState):
    query = state.get("market_task_query", "")
    messages = state.get("messages", [])
    
    modified_state = state.copy()
    if query:
        modified_state["messages"] = messages + [HumanMessage(content=f"TASK: Please perform this specific task: {query}")]
    
    pre_invoke_count = len(modified_state["messages"])
    result = market_agent.invoke(modified_state)
    new_messages = result["messages"][pre_invoke_count:]
    
    report = ""
    for m in reversed(new_messages):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            report = m.content
            break
            
    print(f"DEBUG: Market Report set - length: {len(report)}")
    remaining = [t for t in state.get("remaining_tasks", []) if t != "market"]
    return {"messages": new_messages, "remaining_tasks": remaining, "market_report": report}

def news_agent_node(state: AgentState):
    query = state.get("news_task_query", "")
    messages = state.get("messages", [])
    
    modified_state = state.copy()
    if query:
        modified_state["messages"] = messages + [HumanMessage(content=f"TASK: Please perform this specific task: {query}")]
        
    pre_invoke_count = len(modified_state["messages"])
    result = news_agent.invoke(modified_state)
    new_messages = result["messages"][pre_invoke_count:]
    
    report = ""
    for m in reversed(new_messages):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            report = m.content
            break
            
    print(f"DEBUG: News Report set - length: {len(report)}")
    remaining = [t for t in state.get("remaining_tasks", []) if t != "news"]
    return {"messages": new_messages, "remaining_tasks": remaining, "news_report": report}

def finance_qa_agent_node(state: AgentState):
    query = state.get("finance_qa_task_query", "")
    messages = state.get("messages", [])
    
    modified_state = state.copy()
    if query:
        modified_state["messages"] = messages + [HumanMessage(content=f"TASK: Please perform this specific task: {query}\n\nSTRICT REMINDER: You MUST NOT use general knowledge. If textbooks are empty, you MUST follow the REFUSAL PROTOCOL and explicitly state the limitation.")]
        
    pre_invoke_count = len(modified_state["messages"])
    result = finance_qa_agent.invoke(modified_state)
    new_messages = result["messages"][pre_invoke_count:]
    
    report = ""
    for m in reversed(new_messages):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            report = m.content
            break
            
    # FALLBACK: If education agent is lazy, use general knowledge directly in reports
    if not report and query:
        print("DEBUG: Education agent report empty, attempting internal fallback")
        # Empty report for now, but we'll log it
        
    print(f"DEBUG: Education Report set - length: {len(report)}")
    remaining = [t for t in state.get("remaining_tasks", []) if t != "education"]
    return {"messages": new_messages, "remaining_tasks": remaining, "education_report": report}

def portfolio_agent_node(state: AgentState):
    query = state.get("portfolio_task_query", "")
    messages = state.get("messages", [])
    
    modified_state = state.copy()
    if query:
        modified_state["messages"] = messages + [HumanMessage(content=f"TASK: Please perform this specific portfolio analysis: {query}\n\nSTRICT REMINDER: You MUST NOT use general knowledge. If tools/textbooks are empty, you MUST follow the REFUSAL PROTOCOL and explicitly state the limitation.")]
        
    pre_invoke_count = len(modified_state["messages"])
    result = portfolio_agent.invoke(modified_state)
    new_messages = result["messages"][pre_invoke_count:]
    
    report = ""
    for m in reversed(new_messages):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            report = m.content
            break
            
    print(f"DEBUG: Portfolio Report set - length: {len(report)}")
    remaining = [t for t in state.get("remaining_tasks", []) if t != "portfolio"]
    return {"messages": new_messages, "remaining_tasks": remaining, "portfolio_report": report}

def goal_agent_node(state: AgentState):
    query = state.get("goal_task_query", "")
    messages = state.get("messages", [])
    
    modified_state = state.copy()
    if query:
        modified_state["messages"] = messages + [HumanMessage(content=f"TASK: Please perform this specific goal planning task: {query}\n\nSTRICT REMINDER: You MUST NOT use general knowledge. If tools/textbooks are empty, you MUST follow the REFUSAL PROTOCOL and explicitly state the limitation.")]
        
    pre_invoke_count = len(modified_state["messages"])
    result = goal_agent.invoke(modified_state)
    new_messages = result["messages"][pre_invoke_count:]
    
    report = ""
    for m in reversed(new_messages):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            report = m.content
            break
            
    print(f"DEBUG: Goal Report set - length: {len(report)}")
    remaining = [t for t in state.get("remaining_tasks", []) if t != "goal"]
    return {"messages": new_messages, "remaining_tasks": remaining, "goal_report": report}

def response_synthesizer_node(state: AgentState):
    """
    Combines outputs from multiple agents into a single coherent response.
    """
    market_rep = state.get("market_report", "")
    news_rep = state.get("news_report", "")
    edu_rep = state.get("education_report", "")
    port_rep = state.get("portfolio_report", "")
    goal_rep = state.get("goal_report", "")
    
    print(f"DEBUG: Synthesis phase - M:{bool(market_rep)} N:{bool(news_rep)} E:{bool(edu_rep)} P:{bool(port_rep)} G:{bool(goal_rep)}")
    
    # Native Synthesis Prompt - THE FINAL NARRATOR
    system_prompt = (
        "You are Fintilligence, a high-quality financial AI. Your goal is to provide a single, professional response.\n\n"
        "--- DATA INPUTS ---\n"
        "1. MARKET DATA (Numbers/Prices):\n"
        f"{market_rep or 'No market data available.'}\n\n"
        "2. EDUCATIONAL CONTEXT (Theory/Definitions):\n"
        f"{edu_rep or 'No educational info available.'}\n\n"
        "3. LATEST NEWS:\n"
        f"{news_rep or 'No recent news available.'}\n\n"
        "4. PORTFOLIO ANALYSIS (Strategy/Suggestions):\n"
        f"{port_rep or 'No portfolio analysis available.'}\n\n"
        "5. GOAL PLANNING (Financial Projections/Schedules):\n"
        f"{goal_rep or 'No goal planning data available.'}\n\n"
        "--- MANDATORY GUIDELINES ---\n"
        "1. BE COMPLETE: You MUST answer ALL parts of the query (e.g. both price AND definition).\n"
        "2. NO REPETITION: Don't repeat facts. Merge them gracefully.\n"
        "3. NO WEIRD FORMATTING: Provide a normal, professional paragraph-based response. Do not use vertical text or character separation.\n"
        "4. DISCLAIMER: End with exactly one financial disclaimer.\n\n"
        "Combine the inputs above into a single comprehensive output now:"
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0) 
    response = llm.invoke(system_prompt)
    
    # Tag this message for the UI
    final_msg = AIMessage(
        content=str(response.content),
        additional_kwargs={"is_final_synthesis": True}
    )
    
    return {"messages": [final_msg]}
    
# Router Logic
def router(state: AgentState) -> Literal["market_analyst", "news_synthesizer", "finance_qa_agent", "response_synthesizer", "__end__"]:
    if not state.get("is_financial_query", False):
        return "__end__"
    
    remaining = state.get("remaining_tasks", [])
    if not remaining:
        # User requested individual outputs one after another, bypassing synthesis.
        return "__end__"
    
    next_task = remaining[0]
    if next_task == "market":
        return "market_analyst"
    elif next_task == "news":
        return "news_synthesizer"
    elif next_task == "education":
        return "finance_qa_agent"
    elif next_task == "portfolio":
        return "portfolio_analyst"
    elif next_task == "goal":
        return "goal_planner"
    
    return "__end__"

# Build Graph
builder = StateGraph(AgentState)

builder.add_node("guardrails", guardrails_node)
builder.add_node("market_analyst", market_agent_node)
builder.add_node("news_synthesizer", news_agent_node)
builder.add_node("finance_qa_agent", finance_qa_agent_node)
builder.add_node("portfolio_analyst", portfolio_agent_node)
builder.add_node("goal_planner", goal_agent_node)
builder.add_node("response_synthesizer", response_synthesizer_node)

mapping = {
    "market_analyst": "market_analyst",
    "news_synthesizer": "news_synthesizer",
    "finance_qa_agent": "finance_qa_agent",
    "portfolio_analyst": "portfolio_analyst",
    "goal_planner": "goal_planner",
    "response_synthesizer": "response_synthesizer",
    "__end__": END
}

builder.add_edge(START, "guardrails")
builder.add_conditional_edges("guardrails", router, mapping)
# Loop back to router from agents to check for remaining tasks
builder.add_conditional_edges("market_analyst", router, mapping)
builder.add_conditional_edges("news_synthesizer", router, mapping)
builder.add_conditional_edges("finance_qa_agent", router, mapping)
builder.add_conditional_edges("portfolio_analyst", router, mapping)
builder.add_conditional_edges("goal_planner", router, mapping)
# Synthesizer is the final step
builder.add_edge("response_synthesizer", END)

# Compile with Checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
