import pandas as pd
from ortools.sat.python import cp_model
from datetime import datetime, timedelta

# ---------------------------------------------------------------------
# 1️⃣ Load Data
# ---------------------------------------------------------------------
FILEPATH = "/Users/sivagar/Desktop/sherad/data/combined_jobs_dataset_remerged.xlsx"

print(f"Loading dataset: {FILEPATH}")
df = pd.read_excel(FILEPATH)

# ---------------------------------------------------------------------
# 2️⃣ Basic Cleaning & Filtering
# ---------------------------------------------------------------------
df["DUEDATE"] = pd.to_datetime(df["DUEDATE"], errors="coerce")
df["PLAN_COLOUR"] = df["PLAN_COLOUR"].fillna("White")
df["MACHCODE"] = df["MACHCODE"].fillna("UNKNOWN")

# Filter Göpfert machines only (for Rule 1–4 prototype)
target_machines = ["GPO", "GOP", "GP2", "GO2"]
df = df[df["MACHCODE"].isin(target_machines)].copy()
print(f"Filtered Göpfert jobs: {len(df)}")

# ---------------------------------------------------------------------
# 3️⃣ Horizon Auto-Detect
# ---------------------------------------------------------------------
horizon_start = df["DUEDATE"].min().floor("D")
horizon_end = df["DUEDATE"].max().ceil("D") + pd.Timedelta(days=2)
print(f"Horizon auto-set from {horizon_start} to {horizon_end}")

# ---------------------------------------------------------------------
# 4️⃣ Pre-Compute Durations
# ---------------------------------------------------------------------
feeds_per_min = 60  # 60 feeds/min per Göpfert
df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce").fillna(0)
df["DURATION_MIN"] = (df["QUANTITY"] / feeds_per_min).clip(lower=1).astype(int)

# ---------------------------------------------------------------------
# 5️⃣ Build Scheduling Model
# ---------------------------------------------------------------------
model = cp_model.CpModel()

jobs = []
start_vars, end_vars, interval_vars = {}, {}, {}
machine_to_intervals = {m: [] for m in target_machines}

for i, row in df.iterrows():
    job_id = i
    machine = row["MACHCODE"]
    duration = int(row["DURATION_MIN"])
    colour = row["PLAN_COLOUR"]
    due = row["DUEDATE"]

    # Compute delivery window bounds (Rule 1)
    if colour.lower() == "blue":
        start_window, end_window = 3 * 60, 6 * 60
    elif colour.lower() == "red":
        start_window, end_window = 6 * 60, 8 * 60
    else:  # white/other
        start_window, end_window = 9 * 60, 14 * 60

    horizon_minutes = int((horizon_end - horizon_start).total_seconds() // 60)
    start = model.NewIntVar(0, horizon_minutes, f"start_{job_id}")
    end = model.NewIntVar(0, horizon_minutes, f"end_{job_id}")
    interval = model.NewIntervalVar(start, duration, end, f"interval_{job_id}")

    # Add no-overlap per machine
    machine_to_intervals[machine].append(interval)

    # Soft delivery window constraint (Rule 1)
    violation = model.NewIntVar(0, horizon_minutes, f"viol_{job_id}")
    model.AddMaxEquality(
        violation,
        [
            model.NewIntVarFromDomain(
                cp_model.Domain.FromIntervals([(0, horizon_minutes)]),
                f"temp_low_{job_id}",
            ),
        ],
    )
    # We simulate "distance" from window
    too_early = model.NewIntVar(0, horizon_minutes, f"tooearly_{job_id}")
    too_late = model.NewIntVar(0, horizon_minutes, f"toolate_{job_id}")
    model.Add(too_early >= start_window - end)
    model.Add(too_late >= start - end_window)
    model.AddMaxEquality(violation, [too_early, too_late])

    jobs.append(job_id)
    start_vars[job_id] = start
    end_vars[job_id] = end
    interval_vars[job_id] = interval

# ---------------------------------------------------------------------
# 6️⃣ Machine No-Overlap Constraints
# ---------------------------------------------------------------------
for m, intervals in machine_to_intervals.items():
    if intervals:
        model.AddNoOverlap(intervals)

# ---------------------------------------------------------------------
# 7️⃣ Objective: Minimize Delivery Window Violations
# ---------------------------------------------------------------------
total_violation = model.NewIntVar(0, 1000000, "total_violation")
model.Minimize(total_violation)

# ---------------------------------------------------------------------
# 8️⃣ Solve
# ---------------------------------------------------------------------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60
solver.parameters.num_search_workers = 8

print(f"Solving model... (time limit: {solver.parameters.max_time_in_seconds}s)")
status = solver.Solve(model)

# ---------------------------------------------------------------------
# 9️⃣ Output
# ---------------------------------------------------------------------
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"✅ Solution found: {solver.ObjectiveValue():.2f} total violation")
    schedule = []
    for j in jobs:
        start_min = solver.Value(start_vars[j])
        end_min = solver.Value(end_vars[j])
        start_dt = horizon_start + timedelta(minutes=start_min)
        end_dt = horizon_start + timedelta(minutes=end_min)
        schedule.append(
            {
                "JobID": j,
                "Machine": df.loc[j, "MACHCODE"],
                "Colour": df.loc[j, "PLAN_COLOUR"],
                "Start": start_dt,
                "End": end_dt,
                "Duration_min": df.loc[j, "DURATION_MIN"],
            }
        )
    schedule_df = pd.DataFrame(schedule)
    schedule_df.to_excel("schedule_output.xlsx", index=False)
    print("📁 Schedule saved to schedule_output.xlsx")
else:
    print("❌ No feasible solution found. Status:", solver.StatusName(status))
