# users.py
import streamlit as st
import hashlib

# --- IMPORTANT ---
# This is a mock user database. In a real application, you MUST use a secure, 
# persistent database (e.g., PostgreSQL, Firebase, MongoDB) and proper password hashing.
# For demonstration purposes, we use a simple dictionary stored in session_state.

def initialize_user_db():
    """Initializes the mock user database if it doesn't exist."""
    if 'user_db' not in st.session_state:
        st.session_state.user_db = {
            # "test@example.com": {
            #     "password_hash": "hashed_password", 
            #     "subscription": "none"
            # }
        }

def hash_password(password):
    """Hashes a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def signup_user(email, password):
    """Adds a new user to the mock database."""
    initialize_user_db()
    if email in st.session_state.user_db:
        return False  # User already exists
    
    password_hash = hash_password(password)
    st.session_state.user_db[email] = {
        "password_hash": password_hash,
        "subscription": "none" # Default subscription status
    }
    return True

def login_user(email, password):
    """Authenticates a user against the mock database."""
    initialize_user_db()
    user_data = st.session_state.user_db.get(email)
    if user_data and user_data["password_hash"] == hash_password(password):
        return {"email": email} # Return user info on successful login
    return None
