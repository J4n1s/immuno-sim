import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Antibody Efficacy and Pathology Simulation")
st.write("This app simulates the concentration and affinity of two antibodies, and the resulting disease pathology progression over time.")

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
    conc_model = st.sidebar.selectbox("Concentration model", ["Linear", "Exponential"], key=f"conc_model_{i}", index=1)
    conc_initial = 50.0 if i == 0 else 30.0
    start_conc = st.sidebar.slider("Starting concentration", 0.0, 100.0, conc_initial, key=f"start_conc_{i}")
    if conc_model == "Linear":
        daily_decline = st.sidebar.slider("Daily concentration decline", 0.0, 10.0, 0.4, key=f"conc_decline_{i}")
        half_life = None
    else:
        half_life = st.sidebar.number_input("Concentration half-life (days)", min_value=1, value=50, key=f"half_life_{i}")
        daily_decline = None
    affinity = st.sidebar.slider("Affinity", 0.0, 1.0, 0.3, step=0.01, key=f"affinity_{i}")
    booster = st.sidebar.checkbox("Apply booster shot", key=f"booster_{i}")
    if booster:
        booster_day = st.sidebar.number_input("Booster day (since start)", min_value=0, max_value=days, value=84, step=1, key=f"booster_day_{i}")
        booster_conc = st.sidebar.slider("Booster concentration", 0.0, 100.0, start_conc, key=f"booster_conc_{i}")
        booster_affinity = st.sidebar.slider("Booster affinity", 0.0, 1.0, 0.8, step=0.01, key=f"booster_affinity_{i}")
    else:
        booster_day = None
        booster_conc = None
        booster_affinity = None
    antibodies[ab] = {
        "conc_model": conc_model,
        "start_conc": start_conc / 100.0,
        "daily_decline": (daily_decline / 100.0) if daily_decline is not None else None,
        "half_life": half_life,
        "affinity": affinity,
        "booster": booster,
        "booster_day": booster_day,
        "booster_conc": (booster_conc / 100.0) if booster_conc is not None else None,
        "booster_affinity": booster_affinity
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

# Function to simulate concentration and affinity
def simulate_conc_aff(cfg):
    conc = np.zeros_like(days_array, dtype=float)
    aff = np.zeros_like(days_array, dtype=float)
    current_conc = cfg["start_conc"]
    current_aff = cfg["affinity"]
    if cfg["conc_model"] == "Exponential":
        decay_factor = 0.5 ** (1.0 / cfg["half_life"])
    for t in days_array:
        if t == 0:
            conc[t] = current_conc
            aff[t] = current_aff
        else:
            if cfg["booster"] and t == cfg["booster_day"]:
                current_conc = cfg["booster_conc"]
                current_aff = cfg["booster_affinity"]
                conc[t] = current_conc
                aff[t] = current_aff
                continue
            if cfg["conc_model"] == "Linear":
                current_conc = max(current_conc - cfg["daily_decline"], 0.0)
            else:
                current_conc = current_conc * decay_factor
            conc[t] = current_conc
            aff[t] = current_aff
    return conc, aff

# Run simulations
concentrations = {}
efficiencies = {}
for ab, cfg in antibodies.items():
    conc, aff = simulate_conc_aff(cfg)
    concentrations[ab] = conc
    efficiencies[ab] = conc * aff

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

# Plot concentration
fig0, ax0 = plt.subplots()
for ab, conc in concentrations.items():
    ax0.plot(days_array, conc * 100, label=ab)
ax0.set_xlabel("Day")
ax0.set_ylabel("Concentration")
ax0.legend()
st.pyplot(fig0)

st.text("Efficacy is calculated as concentration times affinity (concentration * affinity).")
# Plot efficacy
fig1, ax1 = plt.subplots()
for ab, eff in efficiencies.items():
    ax1.plot(days_array, eff * 100, label=ab)
ax1.set_xlabel("Day")
ax1.set_ylabel("Efficacy (%)")
ax1.legend()
st.pyplot(fig1)

st.text("Pathology progression is reduced based on antibody efficacy:\n 50% efficacy = 50% of set pathology progression, 10% efficacy = 90% of set pathology progression, etc.")
# Plot pathology
fig2, ax2 = plt.subplots()
for ab, path in dfs.items():
    ax2.plot(days_array, path * 100, label=ab)
ax2.set_xlabel("Day")
ax2.set_ylabel("Pathology (%)")
ax2.legend()
st.pyplot(fig2)

# Final-week delta table
start_fw = days - 6
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
