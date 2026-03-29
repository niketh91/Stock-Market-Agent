import os
import gradio as gr
from agent import run_agent

# ── App config ───────────────────────────────────────────────────
APP_TITLE       = "Stock Market Signal Agent"
APP_DESCRIPTION = """
Ask me about today's stock signals, specific tickers, or correlated pairs.

**Example questions:**
- What signals fired today?
- Tell me about AAPL
- Which tickers move with NVDA?
- Are there any overbought tickers right now?
- Which tickers had a regime change today?
"""

# ── Chat handler ─────────────────────────────────────────────────
def chat(user_message: str, history: list) -> tuple[str, list]:
    """
    Gradio chat handler. Receives the user message and full
    chat history, passes them to the agent, and returns the
    updated history for Gradio to render.
    """
    if not user_message or not user_message.strip():
        return "", history

    try:
        response, updated_history = run_agent(
            user_message = user_message,
            chat_history = history,
        )
        return "", updated_history

    except Exception as e:
        error_msg = f"Something went wrong: {str(e)}"
        history.append((user_message, error_msg))
        return "", history

# ── Gradio UI ────────────────────────────────────────────────────
with gr.Blocks(title=APP_TITLE) as app:

    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(APP_DESCRIPTION)

    chatbot = gr.Chatbot(
        label       = "Stock Signal Agent",
        height      = 500,
        show_copy_button = True,
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder = "Ask about today's signals, a specific ticker, or correlated pairs...",
            label       = "Your question",
            scale       = 9,
            autofocus   = True,
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")

    with gr.Row():
        clear_btn = gr.Button("Clear conversation", variant="secondary")

    gr.Examples(
        examples = [
            "What signals fired today?",
            "Tell me about AAPL",
            "Which tickers move with NVDA?",
            "Are there any overbought tickers right now?",
            "Which tickers had a regime change today?",
            "What are the most volatile tickers today?",
        ],
        inputs = msg_box,
        label  = "Example questions",
    )

    # ── Event handlers ───────────────────────────────────────────
    submit_btn.click(
        fn      = chat,
        inputs  = [msg_box, chatbot],
        outputs = [msg_box, chatbot],
    )

    msg_box.submit(
        fn      = chat,
        inputs  = [msg_box, chatbot],
        outputs = [msg_box, chatbot],
    )

    clear_btn.click(
        fn      = lambda: ([], ""),
        outputs = [chatbot, msg_box],
    )

if __name__ == "__main__":
    app.launch()