import gradio as gr

from app import run_agent


def ask_agent(query: str) -> str:
    # Forward the user query to the smolagents CodeAgent defined in llma.py
    return run_agent(query)


demo = gr.Interface(
    fn=ask_agent,
    inputs=gr.Textbox(label="Enter you Query"),
    outputs=gr.Textbox(label="Answer"),
    title="Simple Research Agent",
    description="Ask anything",
)


if __name__ == "__main__":
    demo.launch()