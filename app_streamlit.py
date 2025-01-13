import asyncio
from pprint import pprint

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph_sdk import get_client

API_URL = "http://127.0.0.1:2024"

client = get_client(url=API_URL)
assistant_id = "obsidian_assistant"


async def get_response(message, history):
    thread = await client.threads.create()

    input = {"messages": history}

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

    try:
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
                            yield (
                                chunk.data[assistant_id]["messages"][-1]["content"],
                                history,
                            )
    except GeneratorExit:
        # Clean up if needed
        print("Generator closed")
        return
    except Exception as e:
        print(f"Error in generator: {e}")
        raise


async def run_conversation(message, history):
    full_response = ""
    async for response, new_history in get_response(message, history):
        full_response = response
        history = new_history
    return full_response, history


# Streamlit App Layout
st.title("Welcome to Obsidian Assistant")
input_text = st.text_input("Ask assistant-related questions here:")

if "history" not in st.session_state:
    st.session_state.history = []

if input_text:
    message_placeholder = st.empty()
    with st.spinner("Processing..."):
        try:
            output, history = asyncio.run(
                run_conversation(input_text, st.session_state.history)
            )
            st.session_state.history = history
            message_placeholder.write(output)
        except Exception as e:
            st.error(f"Error: {e}")
