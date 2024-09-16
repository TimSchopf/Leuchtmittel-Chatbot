import streamlit as st

from llm_agent import LLMAgent

# Custom CSS to style the logo, title, and custom agent icon with responsive design
st.markdown(
    """
    <style>
    .container {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
    }
    .logo-img {
        width: 120px;
        height: 120px;
        margin-left: 20px;
    }
    .logo-text {
        font-weight: 700 !important;
        font-size: 3vw !important;  /* Responsive font size based on viewport width */
        margin: 0;
    }
    .stAvatar {
        width: 50px;
        height: 50px;
    }
    .logo-img {
        width: 110px;
        height: 110px;
        margin-left: 50px;
        object-fit: contain;  /* This ensures the image isn't stretched */
    }
    @media (max-width: 768px) {  /* For tablets and smaller screens */
        .logo-img {
            width: 100px;
            height: 100px;
            margin-left: 20px;
        }
        .logo-text {
            font-size: 7vw !important;  /* Larger font size on smaller screens */
        }
    }
    @media (max-width: 480px) {  /* For smartphones */
        .logo-img {
            width: 100px;
            height: 100px;
            margin-left: 10px;
        }
        .logo-text {
            font-size: 9vw !important;  /* Even larger font size for smartphone screens */
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a container for the title and logo
st.markdown(
    f"""
    <div class="container">
        <h1 class="logo-text">Leuchtmittel Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Brief introduction in German
st.markdown("""
    <div class="intro-text">
    <p>Willkommen bei Ihrem Leuchtmittel Chatbot! Ich bin hier, um Ihnen Auskunft über verschiedene Leuchtmittel zu geben.</p>
    </div>
    """, unsafe_allow_html=True
            )

# Initialize LLMAgent
agent = LLMAgent(openai_api_key=st.secrets["OPENAI_API_KEY"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def reset_conversation():
    st.session_state.messages = []


# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Frage nach Informationen über Leuchtmittel"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("💡 Erleuchtung in Arbeit..."):
            response = agent.chat(st.session_state.messages)
        message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.button('Reset Chat', on_click=reset_conversation)
