import streamlit as st

def load_custom_css():
    """Load custom CSS for premium UI."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        /* --- GLOBAL THEME --- */
        .stApp {
            font-family: 'Outfit', sans-serif !important;
            background-color: #f8fafc !important;
            color: #334155 !important;
        }

        /* --- TYPOGRAPHY --- */
        /* --- TYPOGRAPHY --- */
        .stApp {
            font-family: 'Outfit', sans-serif !important;
        }
        
        h1, h2, h3, h4, h5, h6, p, label, li, .stMarkdown, .stText {
            font-family: 'Outfit', sans-serif !important;
            color: #334155 !important;
        }

        h1, h2, h3 {
            font-weight: 700 !important;
            color: #1e293b !important;
        }

        /* --- STICKY BOTTOM (Fixes Dark Footer) --- */
        [data-testid="stBottom"] {
            background-color: #f8fafc !important;
            border-top: 1px solid #e2e8f0;
            padding-bottom: 2rem !important;
        }

        /* --- INPUT FIELDS (Global) --- */
        .stTextInput input {
            background-color: #ffffff !important;
            color: #1e293b !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
        }
        .stTextInput input:focus {
            border-color: #FF6B00 !important;
            box-shadow: 0 0 0 2px rgba(255, 107, 0, 0.2) !important;
        }

        /* --- CHAT INPUT --- */
        .stChatInput {
            background-color: transparent !important;
        }

        /* The actual chat input box */
        .stChatInput textarea {
            background-color: #ffffff !important;
            color: #1e293b !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 20px !important;
        }

        /* Focus state */
        .stChatInput textarea:focus {
            border-color: #FF6B00 !important;
            box-shadow: 0 0 0 2px rgba(255, 107, 0, 0.2) !important;
        }

        /* --- CHAT MESSAGES --- */
        .stChatMessage {
            background-color: transparent !important;
        }

        /* Bot Message (White Card) */
        [data-testid="stChatMessageContent"] {
            background-color: white !important;
            border: 1px solid #e2e8f0;
            border-radius: 12px !important;
            color: #334155 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        /* User Message (Orange) */
        [data-testid="stChatMessageContent"][class*="user"] {
            background: #FF6B00 !important;
            background: linear-gradient(135deg, #FF6B00, #FF8E53) !important;
            color: white !important;
            border: none !important;
        }

        /* Fix text inside User Bubble */
        [data-testid="stChatMessageContent"][class*="user"] * {
            color: white !important;
        }

        /* --- BUTTONS --- */
        /* Primary (Orange) */
        button[kind="primary"] {
            background-color: #FF6B00 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
        }
        button[kind="primary"] * {
            color: white !important;
        }
        button[kind="primary"]:hover {
            background-color: #e65100 !important;
        }

        /* Secondary (White) */
        .stButton > button {
            background-color: white !important;
            color: #334155 !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
        }
        .stButton > button:hover {
            border-color: #FF6B00 !important;
            color: #FF6B00 !important;
        }

        /* --- EXPANDER (Clean Default) --- */
        .streamlit-expanderHeader {
            background-color: white !important;
            color: #334155 !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
        }
        .streamlit-expanderContent {
            background-color: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px !important;
        }
        
        /* --- STATUS BADGE --- */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            color: #334155;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981; /* Green */
            border-radius: 50%;
            margin-right: 8px;
            box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
        }

        /* --- HIDE ELEMENTS --- */
        #MainMenu, header, footer {
            visibility: hidden;
        }

        /* --- MOBILE --- */
        @media (max-width: 768px) {
            .stApp {
                padding-top: 10px;
            }
        }
        </style>
    """, unsafe_allow_html=True)
