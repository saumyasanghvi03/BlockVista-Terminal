# payments.py
import streamlit as st

def get_subscription_status(email):
    """Retrieves the subscription status for a user from the mock database."""
    if 'user_db' in st.session_state and email in st.session_state.user_db:
        return st.session_state.user_db[email].get("subscription", "none")
    return "none"

def update_subscription_status(email, new_status):
    """Updates the subscription status for a user."""
    if 'user_db' in st.session_state and email in st.session_state.user_db:
        st.session_state.user_db[email]["subscription"] = new_status
        st.session_state.subscription_status = new_status
        st.success("Subscription updated successfully! Please re-run the app.")
        st.rerun()

def display_subscription_plans():
    """Displays subscription plans and a mock payment interface."""
    st.title("Choose Your Plan")
    st.subheader("Unlock the full power of BlockVista Terminal")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.header("Free Tier")
            st.markdown("- **Limited** Dashboard Access")
            st.markdown("- **No Access** to Advanced Features")
            st.markdown("- Broker Integration")
            st.button("Current Plan", disabled=True, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.header("Pro Trader")
            st.subheader("₹499 / month")
            st.markdown("- ✅ **Full** Dashboard Access")
            st.markdown("- ✅ F&O Analytics Hub")
            st.markdown("- ✅ Algo Strategy Hub")
            st.markdown("- ✅ AI Discovery Engine")
            if st.button("Subscribe to Pro", type="primary", use_container_width=True):
                # In a real app, this would redirect to a payment page.
                # Here, we just simulate a successful payment.
                email = st.session_state.bv_user['email']
                update_subscription_status(email, "active")

    with col3:
        with st.container(border=True):
            st.header("Enterprise")
            st.subheader("Contact Us")
            st.markdown("- ✅ All Pro Features")
            st.markdown("- ✅ Dedicated Support")
            st.markdown("- ✅ Custom Integrations")
            st.markdown("- ✅ API Access")
            st.link_button("Contact Sales", "mailto:sales@blockvista.com", use_container_width=True)

    st.warning("This is a mock payment page. Clicking 'Subscribe' will grant you access for this session only.")
