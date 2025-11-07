# streamlit_app.py

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "chandra1024/Chandrakala_AI_Assistant_phi_model"

# ---------------------------
# Load model and tokenizer
# ---------------------------
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model(MODEL_NAME)

# ---------------------------
# Session state for chat history
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Page UI
# ---------------------------
st.set_page_config(page_title="AI Assistant", layout="wide")
st.markdown("<h1 style='text-align:center;color:#4B8BBE;'>AI Assistant 💡</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Limit chat history to last 10 messages
# ---------------------------
if len(st.session_state.history) > 10:
    st.session_state.history = st.session_state.history[-10:]

# ---------------------------
# Display all messages
# ---------------------------
for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["message"])

# ---------------------------
# User input at bottom
# ---------------------------
if user_input := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.history.append({"role": "user", "message": user_input})
    st.chat_message("user").markdown(user_input)

    # ---------------------------
    # Prepare prompt
    # ---------------------------
    system_prompt = """
    You are an AI Assistant built by Chandrakala.
    Your job is to give short, human-like, emotionally warm replies.
    Never add puzzles, stories, or unrelated questions.
    If user asks for a message suggestion, reply with only the message text.
    If the user asks about your base model (questions like 'Which base model are you using?' or 'What model are you built on?'), always reply with: 'Phi-3.
    Never add unrelated examples, hypothetical situations, or extra explanations unless specifically asked.
    If the user asks about deployment on Hugging Face, explain only the actual steps you used.
    """

    last_user = st.session_state.history[-1]["message"]
    last_assistant = ""
    if len(st.session_state.history) >= 2:
        if st.session_state.history[-2]["role"] == "assistant":
            last_assistant = st.session_state.history[-2]["message"]

    text_prompt = f"{system_prompt}\n"
    if last_assistant:
        text_prompt += f"Assistant: {last_assistant}\n"
    text_prompt += f"User: {last_user}\nAssistant:"

    # ---------------------------
    # Generate response
    # ---------------------------
    with st.spinner("Assistant is typing..."):
        inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ---------------------------
    # Extract assistant reply
    # ---------------------------
    assistant_reply = response[len(text_prompt):].strip()
    if "User:" in assistant_reply:
        assistant_reply = assistant_reply.split("User:")[0].strip()

    # Add and display
    st.session_state.history.append({"role": "assistant", "message": assistant_reply})
    st.chat_message("assistant").markdown(assistant_reply)
