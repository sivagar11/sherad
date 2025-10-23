import pandas as pd
from ortools.sat.python import cp_model
from datetime import datetime, timedelta

# =========================================================
# CONFIGURATION
# =========================================================
FILEPATH = "/Users/sivagar/Desktop/sherad/synthetic_jobs_r1to4_full.xlsx"

FEEDS_PER_MIN = 60  # conservative default
R5_GAP_MINUTES = 60  # printed-sheet colour change gap (1 hour)
TARGET_MACHINES_G1 = {"GOP", "GPO"}  # Göpfert 1
TARGET_MACHINES_G2 = {"GO2", "GP2"}  # Göpfert 2
TARGET_MACHINES = list(TARGET_MACHINES_G1 | TARGET_MACHINES_G2)

NOW_OVERRIDE = None  # or set for testing: "2025-06-12 10:00:00"

# weights for soft objectives
W_WINDOW = 1
W_BOARD = 1
W_FREEZE_SOFT = 5
W_BLUE_IMBALANCE = 2

BLUE_IMBALANCE_HARD_CAP = None  # disable hard limit


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def minute_diff(a, b):
    return int((a - b).total_seconds() // 60)


def colour_window_minutes(colour: str):
    c = (colour or "White").strip().lower()
    if c == "blue":
        return 3 * 60, 6 * 60
    elif c == "red":
        return 6 * 60, 8 * 60
    else:
        return 9 * 60, 14 * 60


def infer_processes(row):
    candidates = ["MACHINE1A", "MACHINE1B", "MACHINE2A", "MACHINE2B"]
    present = sum(
        1
        for c in candidates
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != ""
    )
    return present if present > 0 else 1


def hard_force_to_g1(machcode, colcount):
    try:
        cc = int(colcount) if pd.notna(colcount) else None
    except Exception:
        cc = None
    if cc is None or cc < 6:
        return machcode
    if machcode in TARGET_MACHINES_G1:
        return machcode
    if machcode == "GP2":
        return "GPO"
    if machcode == "GO2":
        return "GOP"
    return "GOP"


def safe_sum(model, vars_list, name):
    if vars_list:
        s = model.NewIntVar(0, 10**9, name)
        model.Add(s == sum(vars_list))
        return s
    else:
        z = model.NewIntVar(0, 0, f"{name}_zero")
        model.Add(z == 0)
        return z


# =========================================================
# LOAD DATA
# =========================================================
print(f"Loading dataset: {FILEPATH}")
df = pd.read_excel(FILEPATH)

df["DUEDATE"] = pd.to_datetime(df["DUEDATE"], errors="coerce")
df["PLAN_COLOUR"] = df["PLAN_COLOUR"].fillna("White")
df["MACHCODE"] = df["MACHCODE"].fillna("UNKNOWN")
df["COLCOUNT"] = pd.to_numeric(
    df.get("COLCOUNT", pd.Series([None] * len(df))), errors="coerce"
)

# Filter only Göpfert jobs
df = df[df["MACHCODE"].isin(TARGET_MACHINES)].copy()
print(f"Filtered Göpfert jobs: {len(df)}")

# =========================================================
# R-4 HARD ENFORCEMENT: HIGH COLOUR JOBS ON GÖPFERT 1
# =========================================================
orig_mach = df["MACHCODE"].copy()
df["MACHCODE"] = df.apply(
    lambda r: hard_force_to_g1(r["MACHCODE"], r["COLCOUNT"]), axis=1
)
remapped = (orig_mach != df["MACHCODE"]).sum()
if remapped:
    print(f"R-4: remapped {remapped} high-colour jobs to Göpfert 1 (GOP/GPO).")

# =========================================================
# HORIZON SETUP
# =========================================================
horizon_start = df["DUEDATE"].min().floor("D")
horizon_end = df["DUEDATE"].max().ceil("D") + pd.Timedelta(days=2)
print(f"Horizon auto-set from {horizon_start} to {horizon_end}")
H_MIN = minute_diff(horizon_end, horizon_start)

now = pd.to_datetime(NOW_OVERRIDE) if NOW_OVERRIDE else pd.Timestamp.now()

# =========================================================
# DURATIONS & PRECOMPUTATIONS
# =========================================================
df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce").fillna(0)
df["DURATION_MIN"] = (df["QUANTITY"] / FEEDS_PER_MIN).clip(lower=1).astype(int)

for c in ["MACHINE1A", "MACHINE1B", "MACHINE2A", "MACHINE2B"]:
    if c not in df.columns:
        df[c] = pd.NA
df["PROC_COUNT"] = df.apply(infer_processes, axis=1).clip(lower=1)

df["BOARD_ARRIVAL_USED"] = (
    df["DUEDATE"] - pd.to_timedelta(df["PROC_COUNT"] + 1, unit="D")
).dt.normalize() + pd.Timedelta(hours=17)

if "PLANSTARTDATE" in df.columns:
    df["PLANSTARTDATE"] = pd.to_datetime(df["PLANSTARTDATE"], errors="coerce")
else:
    df["PLANSTARTDATE"] = pd.NaT

# =========================================================
# BUILD CP MODEL
# =========================================================
model = cp_model.CpModel()

start_vars, end_vars, interval_vars = {}, {}, {}
window_viols, board_viols, freeze_soft_devs = [], [], []
blue_g1_lits, blue_g2_lits = [], []
machine_to_intervals = {m: [] for m in TARGET_MACHINES}

for idx, row in df.iterrows():
    job_id = int(idx)
    machine = row["MACHCODE"]
    duration = int(row["DURATION_MIN"])
    colour = str(row["PLAN_COLOUR"])
    colcount = int(row["COLCOUNT"]) if pd.notna(row["COLCOUNT"]) else None

    # Core scheduling vars
    start = model.NewIntVar(0, H_MIN, f"start_{job_id}")
    end = model.NewIntVar(0, H_MIN, f"end_{job_id}")
    interval = model.NewIntervalVar(start, duration, end, f"iv_{job_id}")
    machine_to_intervals[machine].append(interval)

    # R-1: time-window constraint
    start_mod = model.NewIntVar(0, 24 * 60 - 1, f"start_mod_{job_id}")
    model.AddModuloEquality(start_mod, start, 24 * 60)
    win_low, win_high = colour_window_minutes(colour)
    too_early = model.NewIntVar(0, 24 * 60, f"tooEarly_{job_id}")
    too_late = model.NewIntVar(0, 24 * 60, f"tooLate_{job_id}")
    model.Add(too_early >= win_low - (start_mod + duration))
    model.Add(too_late >= start_mod - win_high)
    window_viol = model.NewIntVar(0, 24 * 60, f"winViol_{job_id}")
    model.AddMaxEquality(window_viol, [too_early, too_late])
    window_viols.append(window_viol)

    # R-2: board availability
    ba_dt = row["BOARD_ARRIVAL_USED"]
    if pd.notna(ba_dt):
        earliest_start_min = max(0, minute_diff(ba_dt, horizon_start))
        too_early_board = model.NewIntVar(0, H_MIN, f"boardEarly_{job_id}")
        model.Add(too_early_board >= earliest_start_min - start)
        board_viols.append(too_early_board)

    # R-3: freeze rule
    plan_start = row["PLANSTARTDATE"]
    if pd.notna(plan_start):
        delta_to_plan_s = (plan_start - now).total_seconds()
        planned_min = min(max(0, minute_diff(plan_start, horizon_start)), H_MIN)
        if 0 <= delta_to_plan_s <= 5 * 3600:
            model.Add(start == planned_min)
        elif 0 <= delta_to_plan_s <= 24 * 3600:
            dev = model.NewIntVar(0, H_MIN, f"freezeSoftDev_{job_id}")
            diff = model.NewIntVar(-H_MIN, H_MIN, f"freezeSoftDiff_{job_id}")
            model.Add(diff == start - planned_min)
            model.AddAbsEquality(dev, diff)
            freeze_soft_devs.append(dev)

    # R-4: balance blues
    is_blue = colour.strip().lower() == "blue"
    eligible_blue = is_blue and (colcount is None or colcount <= 5)
    if eligible_blue:
        lit_g1 = model.NewBoolVar(f"blueElig_on_g1_{job_id}")
        lit_g2 = model.NewBoolVar(f"blueElig_on_g2_{job_id}")
        model.Add(lit_g1 == int(machine in TARGET_MACHINES_G1))
        model.Add(lit_g2 == int(machine in TARGET_MACHINES_G2))
        blue_g1_lits.append(lit_g1)
        blue_g2_lits.append(lit_g2)

    # R-5: colour-change gap
    if colcount is not None and colcount >= 5:
        gap_start = end
        gap_end = model.NewIntVar(0, H_MIN, f"gap_end_{job_id}")
        model.Add(gap_end == end + R5_GAP_MINUTES)
        gap_interval = model.NewIntervalVar(
            end, R5_GAP_MINUTES, gap_end, f"gap_{job_id}"
        )
        machine_to_intervals[machine].append(gap_interval)

    start_vars[job_id] = start
    end_vars[job_id] = end
    interval_vars[job_id] = interval

# Machine no-overlap
for m, ivs in machine_to_intervals.items():
    if ivs:
        model.AddNoOverlap(ivs)

# =========================================================
# OBJECTIVE
# =========================================================
total_window = safe_sum(model, window_viols, "total_window")
total_board = safe_sum(model, board_viols, "total_board")
total_freeze_soft = safe_sum(model, freeze_soft_devs, "total_freeze_soft")
blue_g1_count = safe_sum(model, blue_g1_lits, "blue_g1_count")
blue_g2_count = safe_sum(model, blue_g2_lits, "blue_g2_count")

imbalance_diff = model.NewIntVar(-len(df), len(df), "blue_imbalance_diff")
model.Add(imbalance_diff == blue_g1_count - blue_g2_count)
imbalance_abs = model.NewIntVar(0, len(df), "blue_imbalance_abs")
model.AddAbsEquality(imbalance_abs, imbalance_diff)

if BLUE_IMBALANCE_HARD_CAP is not None:
    model.Add(imbalance_abs <= BLUE_IMBALANCE_HARD_CAP)

obj = model.NewIntVar(0, 10**12, "obj")
model.Add(
    obj
    == W_WINDOW * total_window
    + W_BOARD * total_board
    + W_FREEZE_SOFT * total_freeze_soft
    + W_BLUE_IMBALANCE * imbalance_abs
)
model.Minimize(obj)

# =========================================================
# SOLVE
# =========================================================
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60
solver.parameters.num_search_workers = 8

print(f"Solving model... (time limit: {solver.parameters.max_time_in_seconds}s)")
status = solver.Solve(model)

# =========================================================
# OUTPUT
# =========================================================
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(
        f"✅ Solution found: total_window={solver.Value(total_window)} "
        f" total_board={solver.Value(total_board)} "
        f" freeze_soft={solver.Value(total_freeze_soft)} "
        f" blue_imbalance={solver.Value(imbalance_abs)}"
    )

    rows = []
    for j in start_vars.keys():
        s = solver.Value(start_vars[j])
        e = solver.Value(end_vars[j])
        rows.append(
            {
                "ORDERNO": df.loc[j, "ORDERNO"] if "ORDERNO" in df.columns else j,
                "Machine": df.loc[j, "MACHCODE"],
                "PLAN_COLOUR": df.loc[j, "PLAN_COLOUR"],
                "COLCOUNT": df.loc[j, "COLCOUNT"],
                "Start": horizon_start + timedelta(minutes=s),
                "End": horizon_start + timedelta(minutes=e),
                "Duration_min": df.loc[j, "DURATION_MIN"],
                "DUEDATE": df.loc[j, "DUEDATE"],
                "BOARD_ARRIVAL_USED": df.loc[j, "BOARD_ARRIVAL_USED"],
                "PROC_COUNT": df.loc[j, "PROC_COUNT"],
            }
        )

    out = pd.DataFrame(rows).sort_values(["Start", "Machine"])
    out.to_excel("schedule_output_r1to5.xlsx", index=False)
    print("📁 Schedule saved to schedule_output_r1to5.xlsx")

else:
    print("❌ No feasible solution found. Status:", solver.StatusName(status))
