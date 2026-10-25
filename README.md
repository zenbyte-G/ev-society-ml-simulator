# EV Society Charging Management — ML Simulation Starter

A local, interactive prototype for a Pune housing-society EV charging management pitch.

## What it demonstrates
- 100-flat / 6-building scenario (editable)
- 100 kVA transformer example with configurable assumed power factor and reserve
- Synthetic non-EV load profile and synthetic EV charging sessions
- Random Forest regression to forecast next 15-minute non-EV load
- Four policies: uncontrolled, capacity-capped first-come-first-served, fixed overnight window, and ML forecast + priority scheduling
- Optional grid-to-DG outage simulation: EV charging pauses during the selected outage interval
- Metrics and CSV downloads

**Important:** This is a learning/pitch prototype, not certified electrical protection or a real charging controller. All generated results are synthetic and must not be presented as measured savings.

## Requirements
- Python 3.10 or newer recommended
- Internet access only to install the Python packages (not needed to run once installed)
- Windows, macOS, or Linux

## Run on Windows
1. Extract this ZIP.
2. Open the extracted `ev_society_ml_starter` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

   If PowerShell blocks activation, use Command Prompt terminal and run:

   ```bat
   .venv\Scripts\activate.bat
   ```

5. Install packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Start the dashboard:

   ```powershell
   python -m streamlit run app.py
   ```

7. Streamlit will show a local URL, usually `http://localhost:8501`. Open it in your browser.

## First demo settings
- Flats: 100
- Buildings: 6
- Transformer: 100 kVA
- Assumed power factor: 0.90
- Capacity reserve: 15%
- EVs: 25
- Outage enabled: yes

These are illustrative assumptions. Replace them with site-verified values before using the model for engineering decisions.

## How the ML component works
The app creates varied synthetic base-load profiles, trains a Random Forest regressor, and tests it on held-out synthetic rows. The model predicts next-slot non-EV demand. The intelligent policy uses the forecast, current load and a 2 kW buffer to estimate headroom, then schedules EVs in order of earliest departure.

The forecast is intentionally simple. It is a starter component to demonstrate end-to-end integration, not evidence that the policy is optimal.

## What to validate next
1. Collect interval-wise building load data (15-minute or finer) from the society, with permission.
2. Confirm transformer rating, actual usable capacity, power factor, phase balance and existing protection settings with a qualified electrical engineer.
3. Collect charger power, EV arrival/departure, required energy and user priority data.
4. Compare all policies using identical scenarios and seeds.
5. Report prediction MAE, peak load, minutes above the defined safety target, delivered energy, completion by departure, fairness and outage behaviour.
6. Keep any real grid/DG changeover logic in certified electrical protection hardware. The app only models outage behaviour; it must not control real equipment.

## Suggested pitch claim
"This is a synthetic-data proof of concept that demonstrates how short-horizon load forecasting and deadline-aware charging scheduling could help coordinate EV charging under a configurable site capacity target. Field data and electrical validation are required before claiming real-world savings or deploying control."
