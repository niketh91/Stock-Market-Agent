import os
import gradio as gr
from stockagent import run_agent

APP_TITLE = "TickerPulse💲"

APP_DESCRIPTION = """
<div style='text-align: center; padding: 8px 0;'>
    <span style='font-size: 1.1em; color: #4A90D9;'>Real-time market intelligence across 15 U.S. equities</span><br>
    <span style='font-size: 0.9em; color: #888;'>Powered by a live Medallion data pipeline — Bronze, Silver, and Gold — refreshed daily after market close</span>
</div>
"""

FOOTER = """
<div style='text-align: center; padding: 16px 0 4px 0; font-size: 0.82em; color: #888; border-top: 1px solid #e0e0e0; margin-top: 16px;'>
    TickerPulse &nbsp;·&nbsp; 2026 Databricks Winter Data Engineering Bootcamp
    &nbsp;·&nbsp; Powered by Meta Llama 3.3 70B Instruct
    &nbsp;·&nbsp; Built by Sriniketh Muralikrishna
</div>
"""

# Gradio 5.x ChatInterface manages history internally
# Function only needs to accept message and history,
# and return just the response string
def chat(user_message: str, history: list) -> str:
    if not user_message or not user_message.strip():
        return "Please ask a question."
    
    # Debug — print exactly what Gradio passes
    print(f"user_message: {user_message}")
    print(f"history type: {type(history)}")
    print(f"history length: {len(history)}")
    print(f"history content: {history}")

    try:
        response, _ = run_agent(
            user_message = user_message,
            chat_history = history,
        )
        return response
    except Exception as e:
        return f"Something went wrong: {str(e)}"

demo = gr.ChatInterface(
    fn       = chat,
    title    = APP_TITLE,
    description = APP_DESCRIPTION,
    examples = [
        "What signals fired today?",
        "Tell me about AAPL",
        "Which tickers move with NVDA?",
        "Are there any overbought tickers right now?",
        "Which tickers had a regime change today?",
        "What are the most volatile tickers today?",
        "Are any tickers showing a strong buy signal?",
        "Which pairs of stocks are moving together today?",
    ],
)

with demo:
    gr.HTML(FOOTER)

if __name__ == "__main__":
    demo.launch(
        # server_name = "0.0.0.0",
        # server_port = 8080,
        share       = True,
        allowed_paths = ["."]
    )