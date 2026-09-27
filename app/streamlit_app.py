"""
Flight Complexity & Operational Risk Operations Dashboard.
Interactive Streamlit application for airline station managers, dispatchers, and executive leadership.
Enables real-time pre-departure risk prediction, explainable SHAP diagnostics, route hotspot clustering,
and automated frontline operational playbooks.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    ENRICHED_COMPLEXITY_PATH,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
)

# Page configuration
st.set_page_config(
    page_title="SkyOps | Flight Complexity & Risk Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for airline operations aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F7FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-critical {
        background-color: #FED7D7;
        color: #9B2C2C;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-moderate {
        background-color: #FEEBC8;
        color: #9C4221;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #C6F6D5;
        color: #22543D;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv(ENRICHED_COMPLEXITY_PATH)
    return df

@st.cache_resource
def load_models():
    classifier_path = MODELS_DIR / "best_complexity_classifier.joblib"
    pre_dep_path = MODELS_DIR / "pre_departure_model.joblib"
    kmeans_path = MODELS_DIR / "kmeans_routes.joblib"

    classifier = joblib.load(classifier_path) if classifier_path.exists() else None
    pre_dep = joblib.load(pre_dep_path) if pre_dep_path.exists() else None
    kmeans = joblib.load(kmeans_path) if kmeans_path.exists() else None

    return classifier, pre_dep, kmeans

df = load_data()
classifier_meta, pre_dep_meta, kmeans_meta = load_models()

# Sidebar Navigation
st.sidebar.title("✈️ SkyOps AI")
st.sidebar.caption("Flight Complexity & Risk System")
menu = st.sidebar.radio(
    "Navigation",
    [
        "Executive & Fleet Overview",
        "Pre-Departure Risk Simulator",
        "SHAP Explainability & Weights",
        "Route Risk & Network Clusters",
        "Frontline Operations Playbook",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project Origin:** United Airlines Challenge\n"
    "**Version:** Production Enterprise 2.0\n"
    "**Stack:** LightGBM, XGBoost, TreeSHAP, Scikit-Learn"
)

# -------------------------------------------------------------
# Module 1: Executive & Fleet Overview
# -------------------------------------------------------------
if menu == "Executive & Fleet Overview":
    st.markdown('<div class="main-header">✈️ Executive Operations Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fleet-wide complexity monitoring and operational performance across 8,099 scheduled flights.</div>', unsafe_allow_html=True)

    # Top KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    total_flights = len(df)
    avg_complexity = df["overall_flight_complexity"].mean()
    delayed_pct = (df["delayed_15min_flag"].mean()) * 100
    hard_pct = (df["high_complexity_flag"].mean()) * 100
    total_ssr = df["ssr_total"].sum()

    col1.metric("Total Scheduled Flights", f"{total_flights:,}")
    col2.metric("Mean Complexity Score", f"{avg_complexity:.3f}")
    col3.metric("FAA Delayed (>15m) Rate", f"{delayed_pct:.1f}%")
    col4.metric("High Complexity ('Hard')", f"{hard_pct:.1f}%", delta=f"{df['high_complexity_flag'].sum()} flights", delta_color="inverse")
    col5.metric("Special Service (SSR)", f"{total_ssr:,}")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("📊 Flight Complexity Distribution")
        fig_dist = px.histogram(
            df,
            x="overall_flight_complexity",
            color="complexity_class",
            color_discrete_map={"Easy": "#2ECC71", "Medium": "#F39C12", "Hard": "#E74C3C"},
            nbins=35,
            title="Distribution of Flight Complexity Tiers",
            labels={"overall_flight_complexity": "Overall Complexity Score (0 to 1)", "count": "Flight Count"},
        )
        fig_dist.update_layout(template="plotly_white", legend_title="Complexity Tier")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_chart2:
        st.subheader("🏢 Complexity Breakdown by Airline Operator")
        comp_agg = pd.crosstab(df["company_id"], df["complexity_class"], normalize="index") * 100
        comp_agg = comp_agg.reset_index()
        fig_comp = px.bar(
            comp_agg,
            x="company_id",
            y=["Easy", "Medium", "Hard"],
            color_discrete_map={"Easy": "#2ECC71", "Medium": "#F39C12", "Hard": "#E74C3C"},
            title="Carrier Fleet Complexity Composition (%)",
            labels={"company_id": "Airline Carrier Code", "value": "Percentage (%)"},
        )
        fig_comp.update_layout(template="plotly_white", barmode="stack", legend_title="Tier")
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Operations Fleet Table")
    st.dataframe(
        df[["company_id", "flight_number", "route", "scheduled_departure_date_local", "total_seats", "total_pax", "total_bags", "buffer_minutes", "ssr_total", "overall_flight_complexity", "complexity_class"]]
        .head(100),
        use_container_width=True,
    )

# -------------------------------------------------------------
# Module 2: Pre-Departure Risk Simulator
# -------------------------------------------------------------
elif menu == "Pre-Departure Risk Simulator":
    st.markdown('<div class="main-header">🔮 Pre-Departure Flight Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Forecast delay and turnaround failure risk <b>prior to departure</b> using pre-flight passenger, bag, and schedule indicators.</div>', unsafe_allow_html=True)

    if pre_dep_meta is None:
        st.warning("Pre-departure model artifact not found. Please train models first.")
    else:
        pipeline = pre_dep_meta["pipeline"]
        features_meta = pre_dep_meta["features"]

        col_input1, col_input2, col_input3 = st.columns(3)

        with col_input1:
            st.markdown("#### 🛫 Flight & Aircraft")
            sel_company = st.selectbox("Airline Carrier", ["UA", "OO", "G7", "AX", "ZW", "CP"])
            sel_fleet = st.selectbox("Fleet Aircraft Type", ["B737-800", "B777-200", "A320", "ERJ-175", "CRJ-200"])
            sel_seats = st.slider("Total Aircraft Seats", 50, 400, 160)
            sel_dep_hour = st.slider("Scheduled Departure Hour (Local)", 0, 23, 14)
            sel_weekend = st.radio("Weekend Departure?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

        with col_input2:
            st.markdown("#### ⏱️ Turnaround Buffer")
            sel_sched_turn = st.slider("Scheduled Ground Time (mins)", 30, 240, 55)
            sel_min_turn = st.slider("Minimum Required Turn Time (mins)", 25, 120, 45)
            calc_buffer = sel_sched_turn - sel_min_turn
            calc_tightness = sel_min_turn / sel_sched_turn if sel_sched_turn > 0 else 1.0
            st.info(f"Operational Buffer: **{calc_buffer} mins** | Tightness: **{calc_tightness:.2f}**")
            sel_intl = st.checkbox("International Flight?", value=False)

        with col_input3:
            st.markdown("#### 👥 Passengers, Bags & SSR")
            sel_pax = st.slider("Booked Passengers", 10, sel_seats, int(sel_seats * 0.88))
            sel_bags = st.slider("Total Checked Bags", 0, 400, int(sel_pax * 1.1))
            sel_transfer_ratio = st.slider("Baggage Transfer Ratio", 0.0, 1.0, 0.45)
            sel_hot_ratio = st.slider("Hot-Transfer Bag Ratio (<45m conn)", 0.0, 0.8, 0.15)
            sel_wheelchairs = st.number_input("Airport Wheelchairs (SSR)", 0, 50, 5)
            sel_minors = st.number_input("Unaccompanied Minors (SSR)", 0, 20, 1)

        # Build feature dictionary for simulation
        sim_data = {
            "total_seats": [sel_seats],
            "scheduled_ground_time_minutes": [sel_sched_turn],
            "minimum_turn_minutes": [sel_min_turn],
            "buffer_minutes": [calc_buffer],
            "turnaround_tightness_ratio": [calc_tightness],
            "total_pax": [sel_pax],
            "load_factor": [min(sel_pax / sel_seats, 1.5)],
            "child_ratio": [0.08],
            "basic_economy_ratio": [0.15],
            "stroller_ratio": [0.03],
            "total_bags": [sel_bags],
            "bags_per_pax": [sel_bags / sel_pax if sel_pax > 0 else 0.0],
            "transfer_ratio": [sel_transfer_ratio],
            "hot_transfer_ratio": [sel_hot_ratio],
            "num_hot_transfer_bags": [int(sel_bags * sel_hot_ratio)],
            "ssr_total": [sel_wheelchairs + sel_minors],
            "ssr_airport_wheelchair": [sel_wheelchairs],
            "ssr_unaccompanied_minor": [sel_minors],
            "origin_avg_delay": [14.5],
            "dest_avg_delay": [12.0],
            "route_flight_count": [45],
            "scheduled_dep_hour": [sel_dep_hour],
            "scheduled_dep_dayofweek": [5 if sel_weekend else 2],
            "is_weekend": [sel_weekend],
            "is_international": [int(sel_intl)],
            "company_id": [sel_company],
            "carrier": ["Mainline" if sel_seats > 100 else "Express"],
            "fleet_type": [sel_fleet],
            "departure_time_of_day": ["Afternoon" if 12 <= sel_dep_hour < 17 else "Morning"],
        }
        sim_df = pd.DataFrame(sim_data)

        # Run model inference
        pred_prob = pipeline.predict_proba(sim_df)[0, 1]
        decision_threshold = st.slider("Dispatcher Alert Sensitivity Threshold", 0.2, 0.8, 0.40, 0.05,
                                       help="Lower threshold = Higher Recall (catches more delayed flights ahead of time)")

        st.markdown("---")
        st.subheader("🎯 Real-Time Operational Risk Assessment")
        col_res1, col_res2 = st.columns([1, 1.5])

        with col_res1:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_prob * 100,
                title={"text": "Pre-Departure Delay Risk Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1A365D"},
                    "steps": [
                        {"range": [0, 35], "color": "#C6F6D5"},
                        {"range": [35, 60], "color": "#FEEBC8"},
                        {"range": [60, 100], "color": "#FED7D7"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": decision_threshold * 100,
                    },
                },
            ))
            gauge_fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_res2:
            st.markdown("#### Tactical Operations Recommendation")
            if pred_prob >= decision_threshold:
                st.error(f"🚨 **HIGH OPERATIONAL RISK DETECTED** (Probability: {pred_prob:.1%})")
                st.markdown("""
                **Immediate Dispatch Mitigation Protocols:**
                - ⚠️ **Ramp Squeeze:** Turnaround buffer is under pressure. Request priority ramp lead assignment.
                - 🧳 **Hot Baggage Expedite:** Assign dedicated transfer runner cart for connecting bags.
                - ♿ **SSR Wheelchair Escorts:** Pre-position 2 ground agents at gate 15 minutes before scheduled boarding.
                - 📻 **ATC Priority:** Coordinate with Station Control Center for prioritized pushback clearance.
                """)
            else:
                st.success(f"✅ **NORMAL OPERATIONAL PROFILE** (Probability: {pred_prob:.1%})")
                st.markdown("""
                - Operational cushion is adequate.
                - Standard boarding procedures authorized.
                - Nominal baggage loading sequence.
                """)

# -------------------------------------------------------------
# Module 3: SHAP Explainability & Weights
# -------------------------------------------------------------
elif menu == "SHAP Explainability & Weights":
    st.markdown('<div class="main-header">📊 Model Explainability & Weight Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Validating data-driven operational drivers against heuristic hand-picked scoring rules using TreeSHAP.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Heuristic vs Data-Driven Weights", "Global SHAP Drivers", "Local Flight Case Studies"])

    with tab1:
        st.subheader("⚖️ Challenging the Heuristic 25/50/25 Rule")
        st.write(
            "In early competition iterations, arbitrary weights (25% Delay, 50% Passenger, 25% Baggage) were assigned by intuition. "
            "Using TreeSHAP on 8,099 flights, we scientifically validated the true empirical feature contributions."
        )
        weights_img = FIGURES_DIR / "data_driven_vs_heuristic_weights.png"
        if weights_img.exists():
            st.image(str(weights_img), caption="Comparison of Heuristic Weights vs Learned TreeSHAP Attribution", use_column_width=True)

    with tab2:
        st.subheader("🐝 Global SHAP Beeswarm & Top Operational Factors")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            bar_img = FIGURES_DIR / "shap_feature_importance_bar.png"
            if bar_img.exists():
                st.image(str(bar_img), caption="Top 20 Operational Drivers", use_column_width=True)
        with col_s2:
            beeswarm_img = FIGURES_DIR / "shap_summary_beeswarm.png"
            if beeswarm_img.exists():
                st.image(str(beeswarm_img), caption="TreeSHAP Beeswarm Plot (Directional Impact on High Complexity)", use_column_width=True)

    with tab3:
        st.subheader("🔍 Local Flight Waterfall Explanations")
        st.write("Frontline dispatchers can inspect exact feature push/pull forces for individual flights:")
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            img_easy = FIGURES_DIR / "shap_waterfall_easy.png"
            if img_easy.exists():
                st.image(str(img_easy), caption="Easy Flight Breakdown", use_column_width=True)
        with col_w2:
            img_med = FIGURES_DIR / "shap_waterfall_medium.png"
            if img_med.exists():
                st.image(str(img_med), caption="Medium Flight Breakdown", use_column_width=True)
        with col_w3:
            img_hard = FIGURES_DIR / "shap_waterfall_hard.png"
            if img_hard.exists():
                st.image(str(img_hard), caption="High Complexity Flight Breakdown", use_column_width=True)

# -------------------------------------------------------------
# Module 4: Route Risk & Network Clusters
# -------------------------------------------------------------
elif menu == "Route Risk & Network Clusters":
    st.markdown('<div class="main-header">🌐 Route Network & Station Clustering</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unsupervised K-Means clustering and statistical hypothesis testing of airline routes.</div>', unsafe_allow_html=True)

    col_cl1, col_cl2 = st.columns([1.3, 1])
    with col_cl1:
        cluster_img = FIGURES_DIR / "route_clusters_kmeans.png"
        if cluster_img.exists():
            st.image(str(cluster_img), caption="Unsupervised Route Clustering (k=3 Archetypes)", use_column_width=True)

    with col_cl2:
        st.markdown("#### 🧪 Statistical Hypothesis Testing")
        stats_file = REPORTS_DIR / "statistical_testing_summary.json"
        if stats_file.exists():
            import json
            with open(stats_file) as f:
                stats_data = json.load(f)

            anova = stats_data.get("anova_airline_complexity", {})
            chi2 = stats_data.get("chi_square_airline_vs_class", {})

            st.markdown(f"""
            - **One-Way ANOVA (Carrier Variance):**
              - **F-statistic:** `{anova.get('f_statistic', 'N/A')}`
              - **p-value:** `{anova.get('p_value', 0):.2e}` (Significant: {anova.get('statistically_significant')})
              - **Effect Size ($\eta^2$):** `{anova.get('eta_squared_effect_size', 'N/A')}`
            - **Chi-Square Independence (Carrier vs Class):**
              - **$\chi^2$ Statistic:** `{chi2.get('chi2_statistic', 'N/A')}`
              - **p-value:** `{chi2.get('p_value', 0):.2e}`
              - **Cramér's V:** `{chi2.get('cramers_v', 'N/A')}`
            """)

    st.markdown("---")
    st.subheader("🚨 Top 10 Most Complex Routes in Network")
    routes_csv = REPORTS_DIR / "route_complexity_rankings.csv"
    if routes_csv.exists():
        routes_df = pd.read_csv(routes_csv)
        st.dataframe(
            routes_df[["route", "flight_volume", "mean_complexity", "mean_dep_delay", "mean_buffer", "mean_hot_transfer_ratio", "cluster_name"]].head(10),
            use_container_width=True,
        )

# -------------------------------------------------------------
# Module 5: Frontline Operations Playbook
# -------------------------------------------------------------
elif menu == "Frontline Operations Playbook":
    st.markdown('<div class="main-header">📋 Frontline Operations Playbook</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Actionable operational standard operating procedures (SOPs) based on machine learning complexity predictions.</div>', unsafe_allow_html=True)

    col_pb1, col_pb2, col_pb3 = st.columns(3)

    with col_pb1:
        st.markdown("""
        ### 🟢 Tier 1: Easy / Low Risk
        **Operational Profile:**
        - High turnaround buffer (>25 mins)
        - Low transfer baggage ratio (<20%)
        - Standard passenger load
        
        **Standard Operating Procedure:**
        1. Standard gate agent staffing (1 agent).
        2. Standard baggage handling sequence.
        3. Normal pushback queue prioritization.
        """)

    with col_pb2:
        st.markdown("""
        ### 🟡 Tier 2: Medium Risk
        **Operational Profile:**
        - Moderate ground time utilization (>80%)
        - Significant transfer bags or high pax load
        - Minor historical station congestion
        
        **Standard Operating Procedure:**
        1. Pre-alert gate crew 30 minutes prior to arrival.
        2. Assign dedicated baggage runner for transfer bags.
        3. Initiate pre-boarding announcement 5 minutes early.
        """)

    with col_pb3:
        st.markdown("""
        ### 🔴 Tier 3: Hard / Critical Risk
        **Operational Profile:**
        - Negative buffer / tight turn (<15 mins)
        - Heavy hot-transfer baggage volume (>25%)
        - High SSR passenger density (Wheelchairs/Minors)
        
        **Critical Escalation Procedure:**
        1. **Dual-agent gate staffing:** One dedicated to SSR boarding.
        2. **Hot Cart Staging:** Pre-position baggage transfer tug at gate before block-in.
        3. **Ramp Lead Active Supervision:** Ramp supervisor present at gate during entire turn.
        4. **ATC Slot Lock:** Coordinate with airport dispatch for expedited departure slot.
        """)

st.markdown("---")
st.caption("SkyOps AI System | Designed for United Airlines Competition & Production Flight Operations.")
