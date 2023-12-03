import os

import streamlit as st
from openai import OpenAI


from dotenv import load_dotenv
# Load the OpenAI API key from the environment variable
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# Sidebar
st.sidebar.title("Configuration")


def model_callback():
    st.session_state["model"] = st.session_state["model_selected"]


if "model" not in st.session_state:
    st.session_state["model"] = "gpt-3.5-turbo"

st.session_state.model = st.sidebar.radio(
    "Select OpenAI Model",
    ("gpt-3.5-turbo", "gpt-3.5-turbo-16k"),
    index=0 if st.session_state["model"] == "gpt-3.5-turbo" else 1,
    on_change=model_callback,
    key="model_selected",
)

st.sidebar.markdown(
    f"""
    ### ℹ️ <span style="white-space: pre-line; font-family: Arial; font-size: 14px;">Current model: {st.session_state.model}.</span>
    """,
    unsafe_allow_html=True,
)

# Bot roles and their respective initial messages
bot_roles = {
    "English": {
        "role": "system",
        "content": "You are a friendly assistant",
        "description": "This is a standard ChatGPT model.",
    },
    "Polish bot": {
        "role": "system",
        "content": "You are a friendly bot that speaks only Polish",
        "description": "This is a friendly bot speaking in Polish.",
    },
    "German bot": {
        "role": "system",
        "content": "You are a friendly bot that speaks only German",
        "description": "This is a friendly bot speaking in German.",
    },
    "English Pirate bot": {
        "role": "system",
        "content": "You are a friendly bot that speaks only English Pirate",
        "description": "This is a friendly bot speaking in English Pirate.",
    },
}

def bot_role_callback():
    st.session_state["bot_role"] = st.session_state["bot_role_selected"]
    st.session_state["messages"] = [bot_roles[st.session_state["bot_role"]]]

if "bot_role" not in st.session_state:
    st.session_state["bot_role"] = "English"

st.session_state.bot_role = st.sidebar.radio(
    "Select bot role",
    tuple(bot_roles.keys()),
    index=list(bot_roles.keys()).index(st.session_state["bot_role"]),
    on_change=bot_role_callback,
    key="bot_role_selected"
)

description = bot_roles[st.session_state["bot_role"]]["description"]

st.sidebar.markdown(
    f"""
    ### ℹ️ Description
    <span style="white-space: pre-line; font-family: Arial; font-size: 14px;">{description}</span>
    """,
    unsafe_allow_html=True,
)


# Main App
# https://medium.com/@kavita.kc/streamlit-and-chat-models-f4b80d1be391
# https://github.com/shashankdeshpande/langchain-chatbot
# https://discuss.streamlit.io/t/multipage-streamlit-app-with-user-login/26059/19
# https://github.com/MausamGaurav/Streamlit_Multipage_AWSCognito_User_Authentication_Authorization
# https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py
# https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/
st.title("My Own ChatGPT!🤖")

def reset_messages():
    return [bot_roles[st.session_state["bot_role"]]]

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = reset_messages()


# Display messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# user input
if user_prompt := st.chat_input("Your prompt"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate responses
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        response = client.chat.completions.create(model=st.session_state.model,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
        stream=False)
        #print(response)
        full_response = response.choices[0].message.content.strip()
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
