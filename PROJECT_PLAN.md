# Project plan and data specification

## Project objective
Reduce avoidable EV charging peaks in a housing society by forecasting near-term non-EV load and scheduling charging within an engineer-validated operating limit, while respecting departure deadlines and grid/DG outage behaviour.

## System modules
1. **Synthetic / real data input** — non-EV base load, EV arrival, departure, energy request, charger rating, outage state.
2. **Simulation clock** — 15-minute time steps for a 24-hour period.
3. **Load forecaster** — Random Forest regression for next-slot base load.
4. **Scheduler** — charging headroom calculation and deadline-aware EV allocation.
5. **Policy baselines** — uncontrolled, capped FCFS, fixed overnight, forecast-based.
6. **Metrics** — prediction MAE, peak load, time above safety target, time above rated real power, energy served, fully charged EV count.
7. **Dashboard** — settings, charts, comparisons, downloads.
8. **Validation** — replace synthetic assumptions with permitted, anonymized site data and engineering-approved limits.

## Data fields to collect (only with permission)
### Interval load table
- timestamp
- building_or_meter_id (pseudonymized)
- non_ev_load_kw
- grid_or_dg_state
- optional: voltage, current, power_factor, phase loads

### EV session table
- pseudonymous_session_id
- arrival_timestamp
- departure_deadline
- initial_soc_percent (if available)
- target_soc_percent or energy_required_kwh
- charger_max_kw
- user-declared urgency/priority (optional)

Avoid collecting resident names, flat numbers, phone numbers or other personal identifiers unless strictly necessary and properly authorized.

## Safety constraints
- The 100 kVA nameplate is not the same as a 100 kW operating allowance.
- A qualified electrical engineer must define the usable limit using real site conditions, phase balance, protection settings and utility requirements.
- Software forecasts are uncertain; use conservative margins and fail-safe defaults.
- On grid-to-DG switchover, this prototype pauses EV charging. Actual switching and interlocks must be handled by certified protection/control hardware.
- Never connect this student demo to real chargers or switchgear.

## Evaluation plan
For identical base-load profiles, EV sessions and outage windows, compare:
- uncontrolled charging
- first-come-first-served with a capacity cap
- fixed overnight window
- forecast-based scheduling

Report:
- model MAE on held-out days
- peak total load
- minutes above safety target
- minutes above rated real-power approximation
- total energy delivered / requested
- vehicles meeting requested energy by departure
- fairness (e.g., variation in energy served or missed-deadline rate by arrival cohort)
- performance during outages and after restoration

Do not claim the ML method is better until repeated, controlled comparisons demonstrate it.
