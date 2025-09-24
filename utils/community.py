"""
Community, Social, and Plugin Architecture for BlockVista Terminal
Handles: Leaderboard, Badges, Alerts, Trade Journal, Community Chat, Plugins
"""

import streamlit as st
from typing import List, Dict

# Social features (Stub implementations, extend as needed)
def show_leaderboard(data: List[Dict]):
    st.subheader("Leaderboard")
    for user in data:
        st.write(f"{user['name']}: {user['score']} pts")

def display_badges(badges: List[str]):
    st.subheader("Your Badges")
    st.write(", ".join(badges))

def trade_journal(trades: List[Dict]):
    st.subheader("Trade Journal")
    for trade in trades:
        st.write(trade)

def community_chat(messages: List[Dict]):
    st.subheader("Community Chat")
    for msg in messages:
        st.write(f"{msg['user']}: {msg['text']}")

def send_alert(message: str, user: str):
    # Integrate with email, push, webhook, etc.
    st.info(f"Alert to {user}: {message}")

# Simple plugin architecture
PLUGINS = {}

def register_plugin(name: str, func):
    PLUGINS[name] = func

def run_plugin(name: str, *args, **kwargs):
    if name in PLUGINS:
        return PLUGINS[name](*args, **kwargs)
    else:
        st.error(f"Plugin '{name}' not found.")

# Example plugin registration
def sample_plugin():
    st.write("Sample plugin running!")

register_plugin("sample", sample_plugin)