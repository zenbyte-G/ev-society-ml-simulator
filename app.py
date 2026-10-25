import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="EV Society Charging Simulator", page_icon="⚡", layout="wide")

SLOT_MINUTES = 15
SLOTS_PER_DAY = 96
CHARGER_KW = 3.3

def clock_label(slot):
    mins = slot * SLOT_MINUTES
    return f"{mins // 60:02d}:{mins % 60:02d}"

def make_base_load(day_seed, slots=SLOTS_PER_DAY):
    """Synthetic non-EV society demand profile, in kW. Illustrative only."""
    rng = np.random.default_rng(day_seed)
    values = []
    daily_offset = rng.normal(0, 2.0)
    for s in range(slots):
        hour = s / 4
        # Morning bump, daytime demand, and evening peak.
        morning = 10 * math.exp(-((hour - 8.0) / 2.0) ** 2)
        evening = 28 * math.exp(-((hour - 20.0) / 2.1) ** 2)
        daytime = 8 * math.exp(-((hour - 14.0) / 4.0) ** 2)
        base = 28 + morning + daytime + evening + daily_offset + rng.normal(0, 1.8)
        values.append(max(18.0, base))
    return np.array(values)

def make_training_data(days=180, seed=42):
    """Create varied synthetic load profiles for a demonstration forecasting model."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        load = make_base_load(seed + d * 13)
        weekday = d % 7
        for s in range(1, SLOTS_PER_DAY - 1):
            hour = s / 4
            rows.append({
                "hour_sin": math.sin(2 * math.pi * hour / 24),
                "hour_cos": math.cos(2 * math.pi * hour / 24),
                "weekday": weekday,
                "current_base_kw": load[s],
                "previous_base_kw": load[s - 1],
                "two_slots_ago_kw": load[s - 2],
                "target_next_base_kw": load[s + 1],
            })
    return pd.DataFrame(rows)

FEATURES = [
    "hour_sin", "hour_cos", "weekday",
    "current_base_kw", "previous_base_kw", "two_slots_ago_kw"
]

@st.cache_resource
def train_forecaster():
    data = make_training_data()
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    model.fit(train[FEATURES], train["target_next_base_kw"])
    pred = model.predict(test[FEATURES])
    mae = mean_absolute_error(test["target_next_base_kw"], pred)
    return model, mae, len(data)

def forecast_next(model, base_load, slot, weekday=1):
    prev1 = base_load[max(0, slot - 1)]
    prev2 = base_load[max(0, slot - 2)]
    hour = slot / 4
    x = pd.DataFrame([{
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "weekday": weekday,
        "current_base_kw": base_load[slot],
        "previous_base_kw": prev1,
        "two_slots_ago_kw": prev2,
    }])
    return float(model.predict(x[FEATURES])[0])

def make_ev_sessions(n_evs, seed=12):
    """Synthetic EV arrivals. Each EV has an arrival, departure deadline and energy request."""
    rng = np.random.default_rng(seed)
    evs = []
    for i in range(n_evs):
        # Most arrivals occur in the late afternoon/evening, with some variation.
        arrival_hour = float(np.clip(rng.normal(18.3, 2.1), 14.0, 23.5))
        arrival = int(round(arrival_hour * 4))
        dwell_hours = float(rng.uniform(5.0, 12.0))
        departure = min(95, max(arrival + 4, int(round((arrival_hour + dwell_hours) * 4))))
        energy = float(rng.uniform(5.0, 22.0))
        evs.append({
            "ev_id": f"EV-{i+1:02d}",
            "arrival_slot": arrival,
            "departure_slot": departure,
            "energy_kwh": energy,
            "priority": float(rng.uniform(0.0, 1.0)),
        })
    return evs

def run_policy(policy, base_load, evs, safe_limit_kw, rated_kw, model, outage_start, outage_end):
    n = len(base_load)
    remaining = {ev["ev_id"]: ev["energy_kwh"] for ev in evs}
    charged = {ev["ev_id"]: 0.0 for ev in evs}
    rows = []
    schedule_rows = []

    for slot in range(n):
        base = float(base_load[slot])
        outage = outage_start <= slot < outage_end
        active = [
            ev for ev in evs
            if ev["arrival_slot"] <= slot < ev["departure_slot"]
            and remaining[ev["ev_id"]] > 0.01
        ]
        power_by_ev = {}
        if not outage:
            if policy == "uncontrolled":
                # Deliberately ignores the society capacity limit; baseline only.
                for ev in active:
                    power_by_ev[ev["ev_id"]] = min(CHARGER_KW, remaining[ev["ev_id"]] * 4)
            else:
                if policy == "overnight":
                    allowed = slot >= 23 * 4 or slot < 6 * 4
                    available_kw = max(0.0, safe_limit_kw - base) if allowed else 0.0
                elif policy == "intelligent":
                    predicted_base = forecast_next(model, base_load, slot)
                    # Conservative buffer protects against forecast error.
                    available_kw = max(0.0, safe_limit_kw - max(base, predicted_base) - 2.0)
                else:  # capped_fcfs
                    available_kw = max(0.0, safe_limit_kw - base)

                # A charger is either on at 3.3 kW or off; never exceed remaining capacity.
                # Priority: earliest departure, then larger unmet request.
                if policy == "intelligent":
                    active = sorted(active, key=lambda ev: (ev["departure_slot"], -remaining[ev["ev_id"]]))
                else:
                    active = sorted(active, key=lambda ev: (ev["arrival_slot"], ev["departure_slot"]))
                for ev in active:
                    need_kw = min(CHARGER_KW, remaining[ev["ev_id"]] * 4)
                    if available_kw >= need_kw:
                        power_by_ev[ev["ev_id"]] = need_kw
                        available_kw -= need_kw

        ev_kw = sum(power_by_ev.values())
        for ev in evs:
            eid = ev["ev_id"]
            power = power_by_ev.get(eid, 0.0)
            energy = power * SLOT_MINUTES / 60
            remaining[eid] = max(0.0, remaining[eid] - energy)
            charged[eid] += energy
            if ev["arrival_slot"] <= slot < ev["departure_slot"]:
                schedule_rows.append({
                    "slot": slot, "time": clock_label(slot), "ev_id": eid,
                    "charging_kw": power, "charged_kwh_total": charged[eid],
                    "remaining_kwh": remaining[eid], "outage": outage
                })

        total = base + ev_kw
        rows.append({
            "slot": slot, "time": clock_label(slot), "base_load_kw": base,
            "ev_charging_kw": ev_kw, "total_load_kw": total,
            "safe_limit_kw": safe_limit_kw, "rated_kw": rated_kw,
            "outage": outage, "policy": policy,
            "predicted_next_base_kw": forecast_next(model, base_load, slot) if policy == "intelligent" else np.nan
        })

    summary = pd.DataFrame(rows)
    sched = pd.DataFrame(schedule_rows)
    delivered = sum(charged.values())
    requested = sum(ev["energy_kwh"] for ev in evs)
    completed = sum(1 for ev in evs if charged[ev["ev_id"]] >= ev["energy_kwh"] - 0.05)
    eligible = sum(1 for ev in evs if ev["arrival_slot"] < ev["departure_slot"])
    # Over safe limit is a safety-target violation; distinguish from exceeding the rated kW.
    summary["above_safe_limit"] = summary["total_load_kw"] > safe_limit_kw
    summary["above_rated_limit"] = summary["total_load_kw"] > rated_kw
    metrics = {
        "policy": policy,
        "energy_delivered_kwh": delivered,
        "energy_requested_kwh": requested,
        "energy_served_pct": 100 * delivered / requested if requested else 0,
        "vehicles_fully_charged": completed,
        "vehicles_total": len(evs),
        "safe_limit_violation_minutes": int(summary["above_safe_limit"].sum() * SLOT_MINUTES),
        "rated_limit_violation_minutes": int(summary["above_rated_limit"].sum() * SLOT_MINUTES),
        "peak_load_kw": float(summary["total_load_kw"].max()),
        "outage_slots": int(summary["outage"].sum()),
    }
    return summary, sched, metrics

st.title("⚡ EV Charging Management — Society Simulation")
st.caption("Educational prototype using synthetic data. Not a certified electrical protection or real-world control system.")

with st.sidebar:
    st.header("Society configuration")
    flats = st.number_input("Number of flats", min_value=10, max_value=500, value=100, step=10)
    buildings = st.number_input("Number of buildings", min_value=1, max_value=20, value=6, step=1)
    transformer_kva = st.number_input("Transformer rating (kVA)", min_value=25, max_value=1000, value=100, step=25)
    power_factor = st.slider("Assumed power factor", min_value=0.70, max_value=1.00, value=0.90, step=0.01)
    reserve_pct = st.slider("Capacity reserve (%)", min_value=5, max_value=30, value=15, step=1)
    n_evs = st.slider("EVs plugged in during the day", min_value=5, max_value=60, value=25, step=5)
    seed = st.number_input("Scenario seed", min_value=1, max_value=9999, value=21, step=1)
    st.subheader("Grid outage (optional)")
    outage_enabled = st.checkbox("Simulate grid-to-DG switchover", value=True)
    outage_hour = st.slider("Outage start hour", 0, 23, 19)
    outage_duration = st.slider("Outage duration (15-minute slots)", 1, 8, 2)
    run = st.button("Run / rerun simulation", type="primary", use_container_width=True)

model, model_mae, training_rows = train_forecaster()
rated_kw = transformer_kva * power_factor
safe_limit_kw = rated_kw * (1 - reserve_pct / 100)

if "results" not in st.session_state or run:
    base_load = make_base_load(int(seed))
    evs = make_ev_sessions(int(n_evs), seed=int(seed) + 5)
    outage_start = int(outage_hour * 4) if outage_enabled else -1
    outage_end = min(SLOTS_PER_DAY, outage_start + int(outage_duration)) if outage_enabled else -1
    policies = ["uncontrolled", "capped_fcfs", "overnight", "intelligent"]
    results = {}
    metrics = []
    schedules = {}
    for p in policies:
        curve, sched, met = run_policy(
            p, base_load, evs, safe_limit_kw, rated_kw, model, outage_start, outage_end
        )
        results[p] = curve
        schedules[p] = sched
        metrics.append(met)
    st.session_state["results"] = results
    st.session_state["metrics"] = pd.DataFrame(metrics)
    st.session_state["schedules"] = schedules
    st.session_state["evs"] = pd.DataFrame(evs)
    st.session_state["base_load"] = base_load

results = st.session_state["results"]
metrics_df = st.session_state["metrics"]
schedules = st.session_state["schedules"]
ev_df = st.session_state["evs"]
base_load = st.session_state["base_load"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transformer rated real power*", f"{rated_kw:.1f} kW")
c2.metric("Configured safe operating target", f"{safe_limit_kw:.1f} kW")
c3.metric("Forecast model test MAE", f"{model_mae:.2f} kW")
c4.metric("Synthetic training rows", f"{training_rows:,}")
st.caption("*Approximation: kVA × assumed power factor. Actual allowable loading requires electrical-engineer validation and real site data.")

st.subheader("Compare charging policies")
policy_labels = {
    "uncontrolled": "Uncontrolled (baseline)",
    "capped_fcfs": "Capacity-capped FCFS",
    "overnight": "Fixed overnight window",
    "intelligent": "ML forecast + priority scheduling",
}
selected = st.multiselect(
    "Policies to display",
    options=list(policy_labels.keys()),
    default=list(policy_labels.keys()),
    format_func=lambda x: policy_labels[x]
)
plot_df = pd.concat([results[p] for p in selected], ignore_index=True) if selected else pd.DataFrame()
if not plot_df.empty:
    fig = px.line(plot_df, x="time", y="total_load_kw", color="policy",
                  title="Total society load across the day",
                  labels={"time": "Time", "total_load_kw": "Total load (kW)", "policy": "Policy"})
    fig.add_hline(y=safe_limit_kw, line_dash="dash", annotation_text="Safe operating target")
    fig.add_hline(y=rated_kw, line_dash="dot", annotation_text="Approx. rated real power")
    fig.update_layout(xaxis=dict(tickmode="array", tickvals=[clock_label(s) for s in range(0, 96, 8)]))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Performance metrics")
display_metrics = metrics_df.copy()
display_metrics["policy"] = display_metrics["policy"].map(policy_labels)
display_metrics = display_metrics.rename(columns={
    "policy": "Policy", "energy_delivered_kwh": "Energy delivered (kWh)",
    "energy_requested_kwh": "Energy requested (kWh)", "energy_served_pct": "Energy served (%)",
    "vehicles_fully_charged": "Vehicles fully charged", "vehicles_total": "Vehicles total",
    "safe_limit_violation_minutes": "Minutes above safe target",
    "rated_limit_violation_minutes": "Minutes above rated power", "peak_load_kw": "Peak load (kW)",
    "outage_slots": "Outage slots"
})
st.dataframe(display_metrics.round(2), use_container_width=True, hide_index=True)

st.subheader("Inspect one policy")
chosen = st.selectbox("Policy schedule", list(policy_labels.keys()), format_func=lambda x: policy_labels[x], index=3)
chosen_curve = results[chosen]
left, right = st.columns(2)
with left:
    fig2 = px.area(chosen_curve, x="time", y=["base_load_kw", "ev_charging_kw"],
                   title="Base load and EV charging contribution",
                   labels={"value": "Power (kW)", "time": "Time", "variable": "Load component"})
    st.plotly_chart(fig2, use_container_width=True)
with right:
    fig3 = px.bar(chosen_curve, x="time", y="ev_charging_kw",
                  title="EV charging power by time",
                  labels={"ev_charging_kw": "EV charging (kW)", "time": "Time"})
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Synthetic EV sessions")
st.dataframe(ev_df.assign(
    arrival=ev_df["arrival_slot"].map(clock_label),
    departure=ev_df["departure_slot"].map(clock_label)
).drop(columns=["arrival_slot", "departure_slot"]).rename(columns={
    "ev_id": "EV", "energy_kwh": "Requested energy (kWh)", "priority": "Random priority",
    "arrival": "Arrival", "departure": "Departure"
}).round(2), use_container_width=True, hide_index=True)

st.subheader("Download results")
download_df = pd.concat([results[p] for p in results], ignore_index=True)
st.download_button(
    "Download all policy curves as CSV",
    data=download_df.to_csv(index=False).encode("utf-8"),
    file_name="ev_society_simulation_results.csv",
    mime="text/csv"
)
st.download_button(
    "Download synthetic EV sessions as CSV",
    data=ev_df.to_csv(index=False).encode("utf-8"),
    file_name="synthetic_ev_sessions.csv",
    mime="text/csv"
)

with st.expander("What is ML doing here?"):
    st.write(
        "A Random Forest regressor is trained on synthetic non-EV load profiles to predict the next "
        "15-minute base load. The intelligent policy uses the larger of current and predicted base load, "
        "plus a 2 kW buffer, to calculate charging headroom. It then prioritizes vehicles by earliest "
        "departure. This is a demonstrator, not a proven optimal controller."
    )
with st.expander("Important modelling limitations"):
    st.markdown(
        "- All values and EV sessions are synthetic; the output is not evidence of actual savings.\n"
        "- The current scheduler uses discrete 3.3 kW charger decisions and a simple priority rule.\n"
        "- A 100 kVA transformer does not mean 100 kW of usable capacity. Actual limits depend on power factor, "
        "existing load, phase balance, protection settings, ambient conditions and utility rules.\n"
        "- Outage slots suspend EV charging in the model. A real grid/DG interlock must be designed and certified "
        "by qualified electrical professionals; this software must never operate contactors directly.\n"
        "- Forecast quality is measured on synthetic held-out data. Validate with real historical readings before "
        "making operational claims."
    )
