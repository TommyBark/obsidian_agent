import gradio as gr
from gradio import ChatMessage
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph_sdk import get_client

API_URL = "http://127.0.0.1:2024"

client = get_client(url=API_URL)
assistant_id = "obsidian_assistant"


async def get_response(message, history):
    thread = await client.threads.create()

    input = {"messages": history}

    # Stream
    print("-------------------------------------------------------------------------")
    print("History:", history)

    history_langchain_format = []
    for msg in history:
        if msg["role"] == "user":
            history_langchain_format.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=msg["content"]))
    history_langchain_format.append(HumanMessage(content=message))

    input = {"messages": history_langchain_format}

    # TODO: Message streaming

    # Updates streaming
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id=assistant_id,
        input=input,
        stream_mode="updates",
    ):
        print("Chunk:", chunk)
        if chunk.data and chunk.event != "metadata":
            for key in chunk.data.keys():
                if key == assistant_id:
                    if chunk.data[assistant_id]["messages"][-1]["type"] == "ai":
                        yield ChatMessage(
                            role="assistant",
                            content=chunk.data[assistant_id]["messages"][-1]["content"],
                        )


# Create the UI In Gradio
demo = gr.ChatInterface(
    fn=get_response,
    title="Q&A over Speckle's developer docs",
    examples=[
        ["What is my name?"],
        ["Read a note called Buddhism for me."],
    ],
    theme=gr.themes.Soft(),
    type="messages",
)

demo.launch(share=False)
