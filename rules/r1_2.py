import pandas as pd
from ortools.sat.python import cp_model
from datetime import datetime, timedelta

# =========================================================
# 1️⃣  Load dataset
# =========================================================
FILEPATH = "/Users/sivagar/Desktop/sherad/data/combined_jobs_dataset_remerged.xlsx"
print(f"Loading dataset: {FILEPATH}")
df = pd.read_excel(FILEPATH)

# =========================================================
# 2️⃣  Basic Cleaning & Filtering
# =========================================================
df["DUEDATE"] = pd.to_datetime(df["DUEDATE"], errors="coerce")
df["PLAN_COLOUR"] = df["PLAN_COLOUR"].fillna("White")
df["MACHCODE"] = df["MACHCODE"].fillna("UNKNOWN")
df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce").fillna(0)

# Only Göpfert machines for prototype
target_machines = ["GPO", "GOP", "GP2", "GO2"]
df = df[df["MACHCODE"].isin(target_machines)].copy()
print(f"Filtered Göpfert jobs: {len(df)}")

# =========================================================
# 3️⃣  Horizon setup
# =========================================================
horizon_start = df["DUEDATE"].min().floor("D") - timedelta(days=1)
horizon_end = df["DUEDATE"].max().ceil("D") + timedelta(days=2)
print(f"Horizon auto-set from {horizon_start} to {horizon_end}")
horizon_minutes = int((horizon_end - horizon_start).total_seconds() // 60)

# =========================================================
# 4️⃣  Duration estimates
# =========================================================
feeds_per_min = 60  # conservative default
df["DURATION_MIN"] = (df["QUANTITY"] / feeds_per_min).clip(lower=1).astype(int)

# =========================================================
# 5️⃣  Compute board arrival (Rule 2 preprocessing)
# =========================================================
machine_cols = [c for c in df.columns if str(c).startswith("MACHINE")]


def count_processes(row):
    filled = sum(pd.notna(row[c]) and str(row[c]).strip() != "" for c in machine_cols)
    return max(1, filled)  # at least 1


df["PROCESS_COUNT"] = df.apply(count_processes, axis=1)
df["BOARD_ARRIVAL_USED"] = df["DUEDATE"] - pd.to_timedelta(
    df["PROCESS_COUNT"] + 1, unit="D"
)
print(
    f"R-2: {df['BOARD_ARRIVAL_USED'].notna().sum()} jobs constrained by board-arrival availability."
)

# =========================================================
# 6️⃣  Build model
# =========================================================
model = cp_model.CpModel()
start_vars, end_vars, interval_vars = {}, {}, {}
machine_to_intervals = {m: [] for m in target_machines}
viol_window, viol_board = {}, {}

DAY_MINUTES = 24 * 60

for i, row in df.iterrows():
    job_id = i
    machine = row["MACHCODE"]
    duration = int(row["DURATION_MIN"])
    colour = row["PLAN_COLOUR"]
    due = row["DUEDATE"]
    board_date = row["BOARD_ARRIVAL_USED"]

    # Delivery window bounds (Rule 1)
    if colour.lower() == "blue":
        start_window, end_window = 3 * 60, 6 * 60
    elif colour.lower() == "red":
        start_window, end_window = 6 * 60, 8 * 60
    else:  # white/other
        start_window, end_window = 9 * 60, 14 * 60

    # Core scheduling vars
    start = model.NewIntVar(0, horizon_minutes, f"start_{job_id}")
    end = model.NewIntVar(0, horizon_minutes, f"end_{job_id}")
    interval = model.NewIntervalVar(start, duration, end, f"interval_{job_id}")
    machine_to_intervals[machine].append(interval)

    # ===== Rule 1: delivery window penalty (time of day) =====
    minute_of_day = model.NewIntVar(0, DAY_MINUTES - 1, f"minute_of_day_{job_id}")
    model.AddModuloEquality(minute_of_day, start, DAY_MINUTES)

    too_early = model.NewIntVar(0, DAY_MINUTES, f"tooearly_{job_id}")
    too_late = model.NewIntVar(0, DAY_MINUTES, f"toolate_{job_id}")
    model.Add(too_early >= start_window - minute_of_day)
    model.Add(too_late >= minute_of_day - end_window)

    vw = model.NewIntVar(0, DAY_MINUTES, f"vw_{job_id}")
    model.AddMaxEquality(vw, [too_early, too_late])
    viol_window[job_id] = vw

    # ===== Rule 2: board-arrival penalty =====
    if pd.notna(board_date):
        board_min = int((board_date - horizon_start).total_seconds() // 60)
        vb = model.NewIntVar(0, horizon_minutes, f"vb_{job_id}")
        model.Add(vb >= board_min - start)  # positive penalty if start < board arrival
        viol_board[job_id] = vb
    else:
        viol_board[job_id] = model.NewConstant(0)

    start_vars[job_id] = start
    end_vars[job_id] = end
    interval_vars[job_id] = interval

# =========================================================
# 7️⃣  Machine no-overlap
# =========================================================
for m, intervals in machine_to_intervals.items():
    if intervals:
        model.AddNoOverlap(intervals)

# =========================================================
# 8️⃣  Objective: weighted penalties
# =========================================================
W1, W2 = 10, 3  # weights
total_window = model.NewIntVar(0, 1_000_000, "total_window")
total_board = model.NewIntVar(0, 1_000_000, "total_board")
model.Add(total_window == sum(viol_window.values()))
model.Add(total_board == sum(viol_board.values()))
model.Minimize(W1 * total_window + W2 * total_board)

# =========================================================
# 9️⃣  Solve
# =========================================================
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60
solver.parameters.num_search_workers = 8

print(f"Solving model... (time limit: {solver.parameters.max_time_in_seconds}s)")
status = solver.Solve(model)

# =========================================================
# 🔟  Output
# =========================================================
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(
        f"✅ Solution found: total_window={solver.Value(total_window)}  total_board={solver.Value(total_board)}"
    )

    schedule = []
    for j in start_vars:
        s = solver.Value(start_vars[j])
        e = solver.Value(end_vars[j])
        schedule.append(
            {
                "JobID": j,
                "ORDERNO": df.loc[j, "ORDERNO"],
                "Machine": df.loc[j, "MACHCODE"],
                "Colour": df.loc[j, "PLAN_COLOUR"],
                "Start": horizon_start + timedelta(minutes=s),
                "End": horizon_start + timedelta(minutes=e),
                "Duration_min": df.loc[j, "DURATION_MIN"],
                "Violation_Window": solver.Value(viol_window[j]),
                "Violation_Board": solver.Value(viol_board[j]),
            }
        )

    out = pd.DataFrame(schedule)
    out.to_excel("schedule_output.xlsx", index=False)
    print("📁 Schedule saved to schedule_output.xlsx")

else:
    print(f"❌ No feasible solution found. Status: {solver.StatusName(status)}")
