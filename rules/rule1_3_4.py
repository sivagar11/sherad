import pandas as pd
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import LinearExpr
from datetime import timedelta

# ---------------------------------------------------------------------
# 1) Load Data
# ---------------------------------------------------------------------
FILEPATH = "/Users/sivagar/Desktop/sherad/data/combined_jobs_dataset_remerged.xlsx"
print(f"Loading dataset: {FILEPATH}")
df = pd.read_excel(FILEPATH)

# ---------------------------------------------------------------------
# 2) Cleaning & Filter to Göpferts (R-1..R-4 scope)
# ---------------------------------------------------------------------
df["DUEDATE"] = pd.to_datetime(df["DUEDATE"], errors="coerce")
df["PLANSTARTDATE"] = pd.to_datetime(df.get("PLANSTARTDATE"), errors="coerce")
df["PLAN_COLOUR"] = df["PLAN_COLOUR"].fillna("White")
df["MACHCODE"] = df["MACHCODE"].fillna("UNKNOWN")
df["JOBSTATUSID"] = pd.to_numeric(df.get("JOBSTATUSID"), errors="coerce")

target_machines = ["GPO", "GOP", "GP2", "GO2"]
df = df[df["MACHCODE"].isin(target_machines) | df["MACHCODE"].isna()].copy()

# keep only rows with a due date
df = df[df["DUEDATE"].notna()].copy()
print(f"Filtered Göpfert jobs: {len(df)}")

# ---------------------------------------------------------------------
# 3) Horizon
# ---------------------------------------------------------------------
horizon_start = df["DUEDATE"].min().floor("D")
# allow a buffer of +2 days beyond the latest due date
horizon_end = df["DUEDATE"].max().ceil("D") + pd.Timedelta(days=2)
horizon_minutes = int((horizon_end - horizon_start).total_seconds() // 60)
print(f"Horizon auto-set from {horizon_start} to {horizon_end}")

# Treat horizon_start as “now” for freeze logic (as agreed)
now_ts = horizon_start

# ---------------------------------------------------------------------
# 4) Durations (feeds/min = 60)
# ---------------------------------------------------------------------
FEEDS_PER_MIN = 60
df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce").fillna(0)
df["DURATION_MIN"] = (df["QUANTITY"] / FEEDS_PER_MIN).clip(lower=1).astype(int)


# ---------------------------------------------------------------------
# 5) Rule helpers (windows, colours, families)
# ---------------------------------------------------------------------
def window_bounds_minutes(due_ts, colour_name):
    """Return absolute [win_start, win_end] in minutes from horizon_start."""
    # R-1 windows per your rule:
    c = (colour_name or "").strip().lower()
    if c == "blue":
        w0, w1 = 3 * 60, 6 * 60
    elif c == "red":
        w0, w1 = 6 * 60, 8 * 60
    else:
        w0, w1 = 9 * 60, 14 * 60  # white/others
    day0_min = int((due_ts.normalize() - horizon_start).total_seconds() // 60)
    return day0_min + w0, day0_min + w1


# Blue definition (from data)
df["IS_BLUE"] = (df["JOBSTATUSID"] == 1) | (df["PLAN_COLOUR"].str.lower() == "blue")

G1_SET = {"GOP", "GPO"}  # Göpfert 1 family
G2_SET = {"GO2", "GP2"}  # Göpfert 2 family

# Freeze windows (R-3)
FREEZE_HARD_BUFFER_HRS = 5
FREEZE_SOFT_WINDOW_HRS = 24
df["R3_HARD_FREEZE"] = (
    (df["PLANSTARTDATE"].notna())
    & (df["PLANSTARTDATE"] >= now_ts)
    & (df["PLANSTARTDATE"] - now_ts <= pd.Timedelta(hours=FREEZE_HARD_BUFFER_HRS))
)

df["R3_SOFT_FREEZE"] = (
    (df["PLANSTARTDATE"].notna())
    & (df["PLANSTARTDATE"] - now_ts > pd.Timedelta(hours=FREEZE_HARD_BUFFER_HRS))
    & (df["PLANSTARTDATE"] - now_ts <= pd.Timedelta(hours=FREEZE_SOFT_WINDOW_HRS))
)

# ---------------------------------------------------------------------
# 6) Model + Variables
#   We allow machine reassignment (needed for R-4).
#   For each job j and machine m:
#     - presence x[j,m] in {0,1}
#     - optional interval (start[j,m], duration[j], end[j,m]) only if x[j,m]==1
#   Sum_m x[j,m] == 1
#   NoOverlap per machine uses the optional intervals.
# ---------------------------------------------------------------------
model = cp_model.CpModel()

jobs = list(df.index)
machines = target_machines
machine_to_idx = {m: k for k, m in enumerate(machines)}

x = {}  # assign[j,m] bool
start = {}  # start[j,m]
end = {}  # end[j,m]
interval = {}  # interval[j,m]
duration = {}  # duration[j] (constant int)

# Collect for NoOverlap
machine_intervals = {m: [] for m in machines}

for j in jobs:
    dur = int(df.at[j, "DURATION_MIN"])
    duration[j] = dur
    for m in machines:
        x[j, m] = model.NewBoolVar(f"x_{j}_{m}")
        start[j, m] = model.NewIntVar(0, horizon_minutes, f"start_{j}_{m}")
        end[j, m] = model.NewIntVar(0, horizon_minutes, f"end_{j}_{m}")
        interval[j, m] = model.NewOptionalIntervalVar(
            start[j, m], dur, end[j, m], x[j, m], f"I_{j}_{m}"
        )
        machine_intervals[m].append(interval[j, m])

    # exactly one machine must be chosen
    model.Add(sum(x[j, m] for m in machines) == 1)

# No overlap per machine
for m in machines:
    model.AddNoOverlap(machine_intervals[m])

# ---------------------------------------------------------------------
# 7) Rule 1: Delivery window soft penalties
#   For each job and each machine, compute early/late slack vars gated by x[j,m].
#   job_violation_j >= early/late on the chosen machine.
# ---------------------------------------------------------------------
job_window_violation = {}

for j in jobs:
    due = df.at[j, "DUEDATE"]
    colour = df.at[j, "PLAN_COLOUR"]
    win_lo, win_hi = window_bounds_minutes(due, colour)

    # Per-machine gated slacks
    early_terms = []
    late_terms = []

    for m in machines:
        # early_m >= (win_lo - end[j,m]) when x==1, else 0
        early_m = model.NewIntVar(0, horizon_minutes, f"early_{j}_{m}")
        late_m = model.NewIntVar(0, horizon_minutes, f"late_{j}_{m}")

        # Big-M gating with presence
        M = horizon_minutes
        # early_m >= win_lo - end  - M*(1-x)
        model.Add(early_m >= win_lo - end[j, m] - M * (1 - x[j, m]))
        # early_m <= M * x
        model.Add(early_m <= M * x[j, m])

        # late_m >= start - win_hi - M*(1-x)
        model.Add(late_m >= start[j, m] - win_hi - M * (1 - x[j, m]))
        model.Add(late_m <= M * x[j, m])

        early_terms.append(early_m)
        late_terms.append(late_m)

    # job violation = max( max over m early_m, max over m late_m )
    v = model.NewIntVar(0, horizon_minutes, f"viol_{j}")
    max_early = model.NewIntVar(0, horizon_minutes, f"maxearly_{j}")
    max_late = model.NewIntVar(0, horizon_minutes, f"maxlate_{j}")

    model.AddMaxEquality(max_early, early_terms)
    model.AddMaxEquality(max_late, late_terms)
    model.AddMaxEquality(v, [max_early, max_late])
    job_window_violation[j] = v

# ---------------------------------------------------------------------
# 8) Rule 3: Freeze
#   - Hard freeze (<5h): fix machine and exact planned start.
#   - Soft freeze (<24h): penalize changing planned machine; if planned machine is kept,
#                         also penalize moving start time from PLANSTARTDATE.
# ---------------------------------------------------------------------
W_SOFT_FREEZE_MACHINE = 50  # penalty if not on planned machine
W_SOFT_FREEZE_START = 1  # per-minute deviation when on planned machine

soft_freeze_terms = []

for j in jobs:
    if bool(df.at[j, "R3_HARD_FREEZE"]):
        plan_m = str(df.at[j, "MACHCODE"])
        plan_t = df.at[j, "PLANSTARTDATE"]
        if plan_m in machine_to_idx and pd.notna(plan_t):
            plan_min = int((plan_t - horizon_start).total_seconds() // 60)
            # Fix assignment
            for m in machines:
                model.Add(x[j, m] == int(m == plan_m))
            # Fix start==planned
            model.Add(start[j, plan_m] == plan_min)
        # if planned machine not in scope or time missing, we skip (no made-up data)

    elif bool(df.at[j, "R3_SOFT_FREEZE"]):
        plan_m = str(df.at[j, "MACHCODE"])
        plan_t = df.at[j, "PLANSTARTDATE"]
        if (plan_m in machine_to_idx) and pd.notna(plan_t):
            plan_min = int((plan_t - horizon_start).total_seconds() // 60)

            # machine deviation: 1 - x[j, plan_m]
            mdiff = model.NewIntVar(0, 1, f"r3_mdiff_{j}")
            model.Add(mdiff == 1 - x[j, plan_m])
            if W_SOFT_FREEZE_MACHINE > 0:
                soft_freeze_terms.append(LinearExpr.Term(mdiff, W_SOFT_FREEZE_MACHINE))

            # start deviation only if we keep the planned machine (gated by x[j,plan_m])
            if W_SOFT_FREEZE_START > 0:
                # abs(start[j,plan_m] - plan_min) * x[j,plan_m]
                abs_dev = model.NewIntVar(0, horizon_minutes, f"r3_absstart_{j}")
                model.AddAbsEquality(abs_dev, start[j, plan_m] - plan_min)

                # gate: dev_gated <= abs_dev; dev_gated <= M*x; dev_gated >= abs_dev - M*(1-x)
                M = horizon_minutes
                dev_gated = model.NewIntVar(0, horizon_minutes, f"r3_dev_{j}")
                model.Add(dev_gated <= abs_dev)
                model.Add(dev_gated <= M * x[j, plan_m])
                model.Add(dev_gated >= abs_dev - M * (1 - x[j, plan_m]))
                soft_freeze_terms.append(
                    LinearExpr.Term(dev_gated, W_SOFT_FREEZE_START)
                )

# ---------------------------------------------------------------------
# 9) Rule 4: Balance Blue jobs across G1 vs G2 (soft)
# ---------------------------------------------------------------------
W_BLUE_BALANCE = 20

G1 = {m for m in machines if m in G1_SET}
G2 = {m for m in machines if m in G2_SET}

g1_blue_terms = []
g2_blue_terms = []

for j in jobs:
    if bool(df.at[j, "IS_BLUE"]):
        for m in G1:
            g1_blue_terms.append(x[j, m])
        for m in G2:
            g2_blue_terms.append(x[j, m])

g1_blue = model.NewIntVar(0, len(jobs), "g1_blue")
g2_blue = model.NewIntVar(0, len(jobs), "g2_blue")
model.Add(g1_blue == sum(g1_blue_terms) if g1_blue_terms else 0)
model.Add(g2_blue == sum(g2_blue_terms) if g2_blue_terms else 0)

blue_imbalance = model.NewIntVar(0, len(jobs), "blue_imbalance")
model.AddAbsEquality(blue_imbalance, g1_blue - g2_blue)

# ---------------------------------------------------------------------
# 10) Objective: R-1 window + R-3 soft + R-4 balance
# ---------------------------------------------------------------------
total_window = model.NewIntVar(0, 10**9, "total_window")
model.Add(total_window == sum(job_window_violation[j] for j in jobs))

objective_terms = [total_window]
if W_BLUE_BALANCE > 0:
    objective_terms.append(LinearExpr.Term(blue_imbalance, W_BLUE_BALANCE))
objective_terms.extend(soft_freeze_terms)

model.Minimize(sum(objective_terms))

# ---------------------------------------------------------------------
# 11) Solve
# ---------------------------------------------------------------------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60.0
solver.parameters.num_search_workers = 8

print(f"Solving model... (time limit: {solver.parameters.max_time_in_seconds}s)")
status = solver.Solve(model)

# ---------------------------------------------------------------------
# 12) Output
# ---------------------------------------------------------------------
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(
        f"✅ Solution found: total_window={solver.Value(total_window)} "
        f"blue_imbalance={solver.Value(blue_imbalance)}"
    )
    rows = []
    for j in jobs:
        # find chosen machine
        chosen_m = None
        for m in machines:
            if solver.Value(x[j, m]) == 1:
                chosen_m = m
                s = solver.Value(start[j, m])
                e = solver.Value(end[j, m])
                break
        if chosen_m is None:
            # should not happen due to Sum x == 1, but keep safe
            continue

        start_dt = horizon_start + timedelta(minutes=s)
        end_dt = horizon_start + timedelta(minutes=e)
        rows.append(
            {
                "JobRow": j,
                "ORDERNO": df.at[j, "ORDERNO"] if "ORDERNO" in df.columns else None,
                "Colour": df.at[j, "PLAN_COLOUR"],
                "Blue": bool(df.at[j, "IS_BLUE"]),
                "DueDate": df.at[j, "DUEDATE"],
                "ChosenMachine": chosen_m,
                "Duration_min": duration[j],
                "Start": start_dt,
                "End": end_dt,
                "WinViolation_min": solver.Value(job_window_violation[j]),
                "HardFreeze": bool(df.at[j, "R3_HARD_FREEZE"]),
                "SoftFreeze": bool(df.at[j, "R3_SOFT_FREEZE"]),
            }
        )

    out = (
        pd.DataFrame(rows)
        .sort_values(["Start", "ChosenMachine"])
        .reset_index(drop=True)
    )
    out.to_excel("schedule_output.xlsx", index=False)
    print("📁 Schedule saved to schedule_output.xlsx")
else:
    print("❌ No feasible solution. Status:", solver.StatusName(status))
