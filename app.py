import streamlit as st
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ────────────────────────────────────────────────
# Helper: Get base64 of local images
# ────────────────────────────────────────────────
def get_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please place 1.png and 2.jpg in the app folder.")
        return ""

logo_base64 = get_base64("1.png")
bg_base64 = get_base64("2.jpg")

# ────────────────────────────────────────────────
# Model setup
# ────────────────────────────────────────────────
model_name = "Qwen/Qwen2.5-3B-Instruct"

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"  # remove if too slow / OOM
    )
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# ────────────────────────────────────────────────
# Generate response
# ────────────────────────────────────────────────
def generate_response(user_input, chat_history):
    chat_history.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    chat_history.append({"role": "assistant", "content": response})
    return response, chat_history

# ────────────────────────────────────────────────
# Full CSS with Clear button fix
# ────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Roboto+Mono&display=swap');

    /* Root container */
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        position: relative;
        overflow: hidden;
        min-height: 100vh;
    }}

    .stApp {{
        background: transparent !important;
    }}

    /* Header */
    .header {{
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem 0;
        background: rgba(5, 0, 25, 0.5);
        border-bottom: 1px solid rgba(0, 240, 255, 0.4);
        backdrop-filter: blur(6px);
        position: sticky;
        top: 0;
        z-index: 100;
    }}

    .logo {{
        width: 60px;
        height: auto;
        margin-right: 16px;
        filter: drop-shadow(0 0 12px #00eaff);
    }}

    h1 {{
        font-family: 'Orbitron', sans-serif;
        color: #00f0ff;
        text-shadow: 0 0 14px #00f0ff, 0 0 28px #00f0ff77;
        letter-spacing: 3px;
        margin: 0;
        font-size: 2.2rem;
    }}

    /* Glowing chat panels */
    .stChatMessage {{
        border-radius: 12px;
        padding: 12px 18px;
        margin: 10px 10px;
        max-width: 86%;
        backdrop-filter: blur(8px);
        box-shadow: 0 5px 20px rgba(0, 240, 255, 0.25);
        word-wrap: break-word;
    }}

    .stChatMessage.user {{
        background: rgba(0, 80, 140, 0.6);
        border: 1px solid #00d0ff99;
        color: #f0ffff;
        margin-left: auto;
        margin-right: 14px;
    }}

    .stChatMessage.assistant {{
        background: rgba(70, 0, 130, 0.6);
        border: 1px solid #cc66ff99;
        color: #f8efff;
        margin-right: auto;
        margin-left: 14px;
    }}

    /* Input field */
    .stChatInput > div > div {{
        background: rgba(10, 0, 40, 0.75);
        border: 1px solid #00f0ff66;
        border-radius: 12px;
        box-shadow: 0 0 24px rgba(0, 240, 255, 0.28);
        backdrop-filter: blur(12px);
    }}

    .stChatInput input {{
        color: #e0f7ff !important;
    }}

    /* Clear button fix: right-aligned, single line, good spacing */
    .clear-container {{
        display: flex;
        justify-content: flex-end;
        margin-bottom: 0.4rem;
        margin-right: 0.8rem;
    }}

    .stButton > button {{
        min-width: 100px !important;
        font-size: 0.95rem !important;
        padding: 0.5rem 1.2rem !important;
    }}

    /* Footer small */
    .footer {{
        text-align: center;
        padding: 8px 0;
        background: rgba(0, 0, 20, 0.7);
        border-top: 1px solid rgba(204, 102, 255, 0.45);
        color: #b0d0ff;
        font-size: 0.85rem;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 90;
    }}

    /* Drifting stars */
    .stars-layer, .stars-layer2, .stars-layer3 {{
        position: absolute;
        inset: 0;
        pointer-events: none;
        z-index: -1;
    }}

    .stars-layer {{
        background: radial-gradient(circle, white 1px, transparent 1px);
        background-size: 80px 80px;
        animation: drift 160s linear infinite;
        opacity: 0.8;
    }}

    .stars-layer2 {{
        background: radial-gradient(circle, white 1px, transparent 1px);
        background-size: 60px 60px;
        animation: drift 220s linear infinite reverse;
        opacity: 0.6;
    }}

    .stars-layer3 {{
        background: radial-gradient(circle, white 1.3px, transparent 1.3px);
        background-size: 100px 100px;
        animation: drift 300s linear infinite;
        opacity: 0.4;
    }}

    @keyframes drift {{
        from {{ background-position: 0 0; }}
        to   {{ background-position: 1400px 1400px; }}
    }}

    /* Flying rocket */
    .rocket {{
        position: fixed;
        bottom: 100px;
        right: -120px;
        font-size: 48px;
        animation: rocketFly 20s linear infinite;
        z-index: -1;
        opacity: 0.85;
    }}

    @keyframes rocketFly {{
        0%   {{ transform: translateX(0) rotate(-40deg); }}
        100% {{ transform: translateX(-220vw) rotate(-40deg); }}
    }}

    /* Responsive */
    @media (max-width: 768px) {{
        h1 {{ font-size: 1.8rem; }}
        .logo {{ width: 50px; }}
        .stChatMessage {{ max-width: 90%; }}
        .rocket {{ font-size: 40px; bottom: 80px; }}
    }}

    @media (max-width: 480px) {{
        h1 {{ font-size: 1.6rem; }}
        .clear-container {{ margin-right: 0.5rem; }}
    }}

    footer {{visibility: hidden !important;}}
    #MainMenu {{visibility: hidden !important;}}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────
st.markdown(f"""
    <header class="header">
        <img src="data:image/png;base64,{logo_base64}" 
             alt="Alien Robot Logo" class="logo">
        <h1>Alien Chatbot</h1>
    </header>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Chat UI
# ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful alien assistant exploring the cosmos."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ────────────────────────────────────────────────
# Fixed Clear button – right aligned, single line
# ────────────────────────────────────────────────
st.markdown('<div class="clear-container">', unsafe_allow_html=True)
if st.button("Clear", type="primary", key="clear_btn"):
    st.session_state.messages = []
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful alien assistant exploring the cosmos."}
    ]
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Send signal..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            response, st.session_state.chat_history = generate_response(
                prompt, st.session_state.chat_history
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("""
    <footer class="footer">
        Design by Samura Rahman | Powered by Qwen/Qwen2.5-3B-Instruct
    </footer>
""", unsafe_allow_html=True)

# Animated elements
st.markdown("""
    <div class="stars-layer"></div>
    <div class="stars-layer2"></div>
    <div class="stars-layer3"></div>
    <div class="rocket">🚀</div>
""", unsafe_allow_html=True)