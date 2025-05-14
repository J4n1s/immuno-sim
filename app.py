import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Antibody Efficacy and Pathology Simulation")
st.write("This app simulates the efficacy of two antibodies and the resulting disease pathology progression over time.")

# Sidebar for simulation settings
st.sidebar.header("Simulation Settings")
weeks = st.sidebar.number_input("Experiment length (weeks)", min_value=1, max_value=52, value=16, step=1)
days = int(weeks * 7)
st.sidebar.write(f"Total duration: {days} days")

# Sidebar for antibody settings
st.sidebar.header("Antibody Settings")
antibody_names = ["Antibody 1", "Antibody 2"]
antibodies = {}
for i, ab in enumerate(antibody_names):
    st.sidebar.subheader(ab)
    model = st.sidebar.selectbox(f"Model type", ["Linear", "Exponential"], key=f"model_{i}", index=1)
    eff_initial = 50.0 if i == 0 else 30.0
    start_eff = st.sidebar.slider("Starting efficacy (%)", 0.0, 100.0, eff_initial, key=f"start_{i}")
    if model == "Linear":
        daily_decline = st.sidebar.slider("Daily decline (%)", 0.0, 10.0, 0.4, key=f"decline_{i}")
        half_life = None
    else:
        half_life = st.sidebar.number_input("Half-life (days)", min_value=1, value=50, key=f"half_life_{i}")
        daily_decline = None
    booster = st.sidebar.checkbox("Apply booster shot", key=f"booster_{i}")
    if booster:
        booster_day = st.sidebar.number_input("Booster day (since start)", min_value=0, max_value=days, value=84, step=1, key=f"booster_day_{i}")
        booster_mode = st.sidebar.selectbox("Booster mode", ["Set to", "Add"], key=f"booster_mode_{i}")
        booster_value = st.sidebar.slider("Booster value (%)", 0.0, 100.0, start_eff, key=f"booster_value_{i}")
    else:
        booster_day = None
        booster_mode = None
        booster_value = None
    antibodies[ab] = {
        "model": model,
        "start_eff": start_eff / 100.0,
        "daily_decline": (daily_decline / 100.0) if daily_decline is not None else None,
        "half_life": half_life,
        "booster": booster,
        "booster_day": booster_day,
        "booster_mode": booster_mode,
        "booster_value": (booster_value / 100.0) if booster_value is not None else None
    }

# Sidebar for pathology settings
st.sidebar.header("Pathology Settings")
path_model = st.sidebar.selectbox("Pathology model type", ["Linear", "Exponential"], key="path_model", index=1)
path_start = st.sidebar.slider("Starting pathology (%)", 0.0, 100.0, 5.0, key="path_start")
if path_model == "Linear":
    path_daily_inc = st.sidebar.slider("Daily increase (%)", 0.0, 10.0, 1.0, key="path_daily_inc")
    path_rate = None
else:
    path_rate = st.sidebar.slider("Rate (%)", 0.0, 5.0, 3.0, key="path_rate")
    path_daily_inc = None

# Simulation arrays
days_array = np.arange(days + 1)

# Function to simulate antibody efficacy
def simulate_efficacy(cfg):
    eff = np.zeros_like(days_array, dtype=float)
    current = cfg["start_eff"]
    if cfg["model"] == "Exponential":
        decay_factor = 0.5 ** (1.0 / cfg["half_life"])
    for t in days_array:
        if t == 0:
            eff[t] = current
        else:
            # booster at beginning of day
            if cfg["booster"] and t == cfg["booster_day"]:
                if cfg["booster_mode"] == "Set to":
                    current = cfg["booster_value"]
                else:
                    current = min(current + cfg["booster_value"], 1.0)
                eff[t] = current
                continue
            # decay
            if cfg["model"] == "Linear":
                current = max(current - cfg["daily_decline"], 0.0)
            else:
                current = current * decay_factor
            eff[t] = current
    return eff

# Run efficacy simulations
efficiencies = {ab: simulate_efficacy(cfg) for ab, cfg in antibodies.items()}

# Simulate pathology based on efficacy
dfs = {}
for ab, eff in efficiencies.items():
    path = np.zeros_like(days_array, dtype=float)
    path[0] = path_start / 100.0
    for t in range(1, days + 1):
        e = eff[t]
        if path_model == "Linear":
            inc = (path_daily_inc / 100.0) * (1 - e)
            path[t] = path[t - 1] + inc
        else:
            rate = path_rate / 100.0
            path[t] = path[t - 1] * (1 + rate * (1 - e))
    dfs[ab] = path

# Plot efficacy
fig1, ax1 = plt.subplots()
for ab, eff in efficiencies.items():
    ax1.plot(days_array, eff * 100, label=ab)
ax1.set_xlabel("Day")
ax1.set_ylabel("Efficacy (%)")
ax1.legend()
st.pyplot(fig1)

# Plot pathology
fig2, ax2 = plt.subplots()
for ab, path in dfs.items():
    ax2.plot(days_array, path * 100, label=ab)
ax2.set_xlabel("Day")
ax2.set_ylabel("Pathology (%)")
ax2.legend()
st.pyplot(fig2)

# Final-week delta table
start_fw = days - 6  # last 7 days
names = list(dfs.keys())

data = {
    "Day": days_array[start_fw:],
    names[0]: np.round(dfs[names[0]][start_fw:] * 100, 2),
    names[1]: np.round(dfs[names[1]][start_fw:] * 100, 2),
    "Delta (%)": np.round((dfs[names[1]][start_fw:] - dfs[names[0]][start_fw:]) * 100, 2)
}
df_delta = pd.DataFrame(data)

st.subheader("Final Week Pathology Difference")
st.dataframe(df_delta)
