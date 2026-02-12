import streamlit as st
import requests

st.title("Website Chatbot 🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask a question about our Rays of Grace...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    response = requests.post(
        "http://127.0.0.1:8000/chat",
        json={"query": user_input}
    )

    answer = response.json()["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
