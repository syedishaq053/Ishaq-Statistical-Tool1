import streamlit as st
import requests
from datetime import datetime
from supabase import create_client, Client
from typing import Optional

# ---------------------------
# Load Supabase credentials (from secrets)
# ---------------------------
def init_supabase() -> Optional[Client]:
    try:
        supabase_url = st.secrets.get("SUPABASE_URL", "")
        supabase_key = st.secrets.get("SUPABASE_KEY", "")
        if supabase_url and supabase_key:
            return create_client(supabase_url, supabase_key)
    except Exception:
        pass
    return None


# ---------------------------
# Country detection with fallbacks
# ---------------------------
def detect_country() -> str:
    """Return two‑letter country code or 'Unknown'."""
    if "detected_country" in st.session_state:
        return st.session_state["detected_country"]

    endpoints = [
        ("https://ipapi.co/country/", 3),
        ("https://ip-api.com/line/?fields=countryCode", 3),
        ("https://ipinfo.io/country", 3),
    ]
    for url, timeout in endpoints:
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                country = resp.text.strip()
                if len(country) == 2 and country.isalpha() and country.upper() == country:
                    st.session_state["detected_country"] = country
                    return country
        except Exception:
            continue
    st.session_state["detected_country"] = "Unknown"
    return "Unknown"


# ---------------------------
# Registration / trial logic
# ---------------------------
def handle_access():
    """
    Main entry point for access control.
    Returns True if user is allowed to see the app, False if registration wall is shown.
    """
    # Initialize Supabase client
    supabase = init_supabase()

    # Country detection
    if "user_country" not in st.session_state:
        st.session_state.user_country = detect_country()

    country = st.session_state.user_country
    trial_active = st.session_state.get("trial_active", False)

    # ----- Case 1: India (auto‑detected) -----
    if country == "IN":
        st.success("🇮🇳 Free access for India")
        return True

    # ----- Case 2: Already have an active trial -----
    if trial_active:
        return True

    # ----- Case 3: Unknown country -> ask user to pick -----
    if country == "Unknown":
        st.info("🌍 We could not automatically detect your country. Please select your location to start your free trial.")
        selected = st.selectbox(
            "Select your country",
            ["IN", "US", "GB", "CA", "AU", "DE", "FR", "JP", "BR", "Other"],
        )
        if selected == "Other":
            selected = st.text_input("Enter your country code (e.g., IN, US)", value="IN", max_chars=2).upper()
        st.session_state.user_country = selected
        country = selected  # update local variable

    # ----- Registration wall (non‑IN users, no trial) -----
    st.markdown(
        """
<div style="max-width:480px;margin:60px auto;padding:32px 36px;
     background:#f8faff;border:1px solid #c7d2fe;border-radius:14px;
     box-shadow:0 4px 24px rgba(67,97,238,.10);">
  <h2 style="text-align:center;color:#3730a3;margin-bottom:4px;">
    📊 IshaQ‑Krishnan Stats Suite
  </h2>
  <p style="text-align:center;color:#666;font-size:0.9rem;margin-bottom:20px;">
    Comprehensive statistical analysis · Free 90‑day trial for new users
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.info("🌍 Enter your email below to start your **free 90‑day trial**.")
    reg_email = st.text_input("Email address", placeholder="you@example.com", key="reg_email")

    if st.button("🚀 Start Free Trial", type="primary"):
        if not reg_email or "@" not in reg_email:
            st.warning("Please enter a valid email address.")
        elif supabase is None:
            st.success("✅ Trial started (local mode).")
            st.session_state["trial_active"] = True
            st.rerun()
        else:
            try:
                # Check if user already exists
                existing = (
                    supabase.table("users")
                    .select("trial_start,email")
                    .eq("email", reg_email)
                    .execute()
                )
                if not getattr(existing, "data", None):
                    supabase.table("users").insert(
                        {"email": reg_email, "country": country}
                    ).execute()
                    st.success("✅ Trial started! You have 90 days of free access.")
                    st.session_state["trial_active"] = True
                    st.rerun()
                else:
                    trial_start = datetime.fromisoformat(
                        existing.data[0]["trial_start"].replace("Z", "+00:00")
                    )
                    days_used = (datetime.now().astimezone() - trial_start).days
                    if days_used < 90:
                        st.success(f"✅ Welcome back! You have **{90 - days_used} days** left.")
                        st.session_state["trial_active"] = True
                        st.rerun()
                    else:
                        st.error("❌ Your 90‑day trial has ended. Please subscribe to continue.")
            except Exception as e:
                st.error(f"Registration failed: {e}")
    return False   # Stop the app (will be executed after st.stop)
