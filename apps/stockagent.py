import os
import json
import datetime
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from databricks_langchain import ChatDatabricks
from tools import (
    get_latest_date,
    get_latest_signals,
    get_regime_changes,
    get_high_volatility_tickers,
    get_correlated_pairs,
    get_divergence_alert,
)

# ── Config from environment variables ───────────────────────────
LLM_ENDPOINT = os.getenv("DATABRICKS_LLM_ENDPOINT",
                          "databricks-meta-llama-3-3-70b-instruct")

# ── LLM setup ───────────────────────────────────────────────────
llm = ChatDatabricks(
    endpoint    = LLM_ENDPOINT,
    temperature = 0.1,
    max_tokens  = 2000,
)

# ══════════════════════════════════════════════════════════════════
# LANGCHAIN TOOL WRAPPERS
# ══════════════════════════════════════════════════════════════════

@tool
def tool_get_latest_signals(trade_date: str) -> str:
    """
    Returns a snapshot of all tickers for the given trade date.
    Includes RSI, signal regime, volatility rank, volume ratio,
    and best correlated partner for each ticker.
    Always call this first before any other tool.
    """
    df = get_latest_signals(trade_date)
    return df.to_string(index=False) if len(df) > 0 else "No data found."

@tool
def tool_get_regime_changes(trade_date: str) -> str:
    """
    Returns tickers where the signal regime changed on the given
    trade date compared to the previous day.
    Use this to identify tickers with meaningful momentum shifts.
    """
    df = get_regime_changes(trade_date)
    return df.to_string(index=False) if len(df) > 0 else "No regime changes found."

@tool
def tool_get_high_volatility_tickers(trade_date: str,
                                      threshold: float = 0.75) -> str:
    """
    Returns tickers in the top percentile of volatility on the
    given trade date. threshold=0.75 means top 25% most volatile.
    Use this to identify tickers with unusual price movement.
    """
    df = get_high_volatility_tickers(trade_date, threshold)
    return df.to_string(index=False) if len(df) > 0 else "No high volatility tickers found."

@tool
def tool_get_correlated_pairs(ticker: str,
                               min_consistency_pct: float = 30.0) -> str:
    """
    Returns structural correlation partners for a specific ticker.
    Use this after identifying an interesting ticker to understand
    whether its correlated peers show the same signal.
    If peers show the same signal, confidence should be higher.
    """
    df = get_correlated_pairs(ticker, min_consistency_pct)
    return df.to_string(index=False) if len(df) > 0 \
        else f"No structural partners found for {ticker}."

@tool
def tool_get_divergence_alert(trade_date: str,
                               min_consistency_pct: float = 30.0,
                               divergence_threshold: float = 0.2) -> str:
    """
    Returns pairs that historically move together but diverged today.
    Use this to identify CORRELATION_BREAK signals.
    """
    df = get_divergence_alert(trade_date, min_consistency_pct,
                               divergence_threshold)
    return df.to_string(index=False) if len(df) > 0 \
        else "No divergent pairs found."

tools = [
    tool_get_latest_signals,
    tool_get_regime_changes,
    tool_get_high_volatility_tickers,
    tool_get_correlated_pairs,
    tool_get_divergence_alert,
]

tool_map        = {t.name: t for t in tools}
llm_with_tools  = llm.bind_tools(tools)

# ══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """
You are a quantitative stock market analyst monitoring U.S. equity tickers.
Today's date is {trade_date}.

You have access to five tools to query enriched stock data:
- tool_get_latest_signals        : start here for every analysis
- tool_get_regime_changes        : find tickers with momentum shifts
- tool_get_high_volatility_tickers: find unusually volatile tickers
- tool_get_correlated_pairs      : get correlation partners for a ticker
- tool_get_divergence_alert      : find pairs that broke their relationship

You can answer three types of questions:
1. Questions about today's signals  e.g. "what signals fired today?"
2. Questions about specific tickers e.g. "tell me about AAPL"
3. Questions about correlated pairs e.g. "which tickers move with NVDA?"

When generating trading signals, output a JSON array at the end like this:
SIGNALS: [{{"ticker": "AAPL", "signal_type": "MOMENTUM_ALERT",
            "rationale": "reasoning here", "confidence": 0.85,
            "rsi_at_signal": 74.2, "regime_at_signal": "STRONG_SELL"}}]

Valid signal types:
MOMENTUM_ALERT, OVERSOLD_ALERT, REGIME_CHANGE,
VOLATILITY_SPIKE, CORRELATION_BREAK, CLUSTER_MOVE

Rules:
- Only fire a signal if the data genuinely supports it
- Do not force a signal if nothing notable exists
- Always call tool_get_latest_signals first
- When you find an interesting ticker call tool_get_correlated_pairs
  to check if peers show the same pattern
- For conversational questions answer clearly in plain English
- Reference actual numbers from the data in your responses
"""

# ══════════════════════════════════════════════════════════════════
# AGENT LOOP
# ══════════════════════════════════════════════════════════════════

def run_agent(user_message: str,
              chat_history: list,
              max_iterations: int = 10) -> tuple[str, list]:
    """
    Runs the agent loop for a single user message.

    Args:
        user_message  : the user's chat input
        chat_history  : list of previous (user, assistant) tuples
                        from Gradio — maintains conversation context
        max_iterations: safety limit on tool call rounds

    Returns:
        response_text : the agent's final natural language response
        chat_history  : updated history with new turn appended
    """
    trade_date = get_latest_date()

    print(f"run_agent called with: {user_message}")
    print(f"chat_history length: {len(chat_history)}")

    # Build system prompt with today's date injected
    system_prompt = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(trade_date=trade_date)
    )

    messages = [system_prompt]

    # Handle Gradio 5.x dict format {"role": ..., "content": ...}
    # and legacy tuple format (human, assistant)
    for item in chat_history:
        if isinstance(item, dict):
            role    = item.get("role", "")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=content))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            human, assistant = item
            if human:
                messages.append(HumanMessage(content=human))
            if assistant:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=assistant))

    # Add current user message
    messages.append(HumanMessage(content=user_message))

    # Agent loop
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        print(f"\n--- Iteration {iteration} ---")

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        print(f"tool_calls: {response.tool_calls}")
        print(f"content preview: {response.content[:100] if response.content else 'empty'}")

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name   = tool_call["name"]
                tool_args   = tool_call["args"]
                print(f"Calling tool: {tool_name} with {tool_args}")
                tool_fn     = tool_map[tool_name]
                tool_result = tool_fn.invoke(tool_args)

                messages.append(ToolMessage(
                    content     = str(tool_result),
                    tool_call_id= tool_call["id"]
                ))
        else:
            # LLM finished reasoning — extract response
            response_text = response.content
            return response_text, chat_history

    # Safety fallback if max iterations hit
    fallback = "I reached my reasoning limit. Please try a more specific question."
    return fallback, chat_history


# ══════════════════════════════════════════════════════════════════
# SIGNAL WRITING (called separately by the scheduled pipeline)
# ══════════════════════════════════════════════════════════════════

def parse_signals(llm_output: str) -> list:
    """Extracts JSON signal array from LLM output."""
    try:
        start     = llm_output.index("SIGNALS:") + len("SIGNALS:")
        json_str  = llm_output[start:].strip()
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        return []