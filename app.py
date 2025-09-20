# app.py
import streamlit as st
from kiteconnect import KiteConnect
from utils.style import set_blockvista_style
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="BlockVista Login", layout="centered")
set_blockvista_style()

st.title("BlockVista Terminal")

# --- Auto-Refresh Controls ---
if st.session_state.get('kite'): # Only show controls if logged in
    st.sidebar.header("Live Data")
    auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
    refresh_interval = st.sidebar.number_input("Interval (s)", min_value=5, max_value=60, value=5, disabled=not auto_refresh)
    
    if auto_refresh:
        st_autorefresh(interval=refresh_interval * 1000, key="data_refresher")

# --- Kite Connect Login Flow ---
st.subheader("Zerodha Kite Authentication")
try:
    api_key = st.secrets["KITE_API_KEY"]
    api_secret = st.secrets["KITE_API_SECRET"]
except (FileNotFoundError, KeyError):
    st.error("Kite API credentials not found. Please create and populate `.streamlit/secrets.toml`.")
    st.stop()

kite = KiteConnect(api_key=api_key)

if 'access_token' in st.session_state:
    try:
        kite.set_access_token(st.session_state.access_token)
        st.session_state['kite'] = kite
        profile = kite.profile()
        st.success(f"Authenticated as {profile['user_name']}")
        st.page_link("pages/1_Dashboard.py", label="Go to Dashboard", icon="📊")

        if st.sidebar.button("Logout"):
            del st.session_state.access_token
            del st.session_state.kite
            st.rerun()
    except Exception as e:
        st.error(f"Session expired or invalid. Please log in again.")
        del st.session_state.access_token
        if 'kite' in st.session_state: del st.session_state.kite
else:
    request_token = st.query_params.get("request_token")
    if request_token:
        try:
            data = kite.generate_session(request_token, api_secret=api_secret)
            st.session_state.access_token = data["access_token"]
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")
    else:
        st.link_button("Login with Kite", kite.login_url())
