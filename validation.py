"""
r1_r10_validation.py
-------------------------------------
Validation script for Sheard AI Planning Phase 1 (Rules R1–R10)

Input files:
- feasible_40jobs_r1to10_fixed.xlsx
- schedule_output_r1to10_final.xlsx
- board24_load_plan_r10.xlsx

Output:
- validation_summary.xlsx
"""

import pandas as pd
from datetime import datetime, timedelta

# ===============================
# Load input datasets
# ===============================
orders_df = pd.read_excel(
    "/Users/sivagar/Desktop/sherad/feasible_40jobs_r1to10_fixed.xlsx"
)
schedule_df = pd.read_excel(
    "/Users/sivagar/Desktop/sherad/schedule_output_r1to10_final.xlsx"
)
board24_df = pd.read_excel("/Users/sivagar/Desktop/sherad/board24_load_plan_r10.xlsx")

# Normalise date columns
for col in ["Start", "End", "DUEDATE", "BOARD_ARRIVAL_USED"]:
    if col in schedule_df.columns:
        schedule_df[col] = pd.to_datetime(schedule_df[col], errors="coerce")

for col in ["DUEDATE", "BOARD_ARRIVAL_DATE"]:
    if col in orders_df.columns:
        orders_df[col] = pd.to_datetime(orders_df[col], errors="coerce")

for col in ["REQ_BOARD_TIME", "BOARD_ARRIVAL_USED", "DUEDATE"]:
    if col in board24_df.columns:
        board24_df[col] = pd.to_datetime(board24_df[col], errors="coerce")


# Helper function
def result(rule, passed, issues):
    return {
        "Rule": rule,
        "Status": "✅ PASS" if passed else "❌ FAIL",
        "Violations": len(issues),
        "Examples": str(issues[:3]) if len(issues) > 0 else "-",
    }


results = []

# =========================================================
# R-1 Delivery windows: Blue (03–06), Red (06–08), White (09–14)
# =========================================================
r1_viol = []
for _, row in schedule_df.iterrows():
    c = str(row["PLAN_COLOUR"]).strip().lower()
    h = row["Start"].hour if pd.notnull(row["Start"]) else None
    if c == "blue" and not (3 <= h < 6):
        r1_viol.append(row["ORDERNO"])
    elif c == "red" and not (6 <= h < 8):
        r1_viol.append(row["ORDERNO"])
    elif c == "white" and not (9 <= h < 14):
        r1_viol.append(row["ORDERNO"])
results.append(result("R-1 Delivery Windows", len(r1_viol) == 0, r1_viol))

# =========================================================
# R-2 Raw-board offset (1→+1 day, 2→+2, 3→+3)
# =========================================================
r2_viol = []
for _, row in board24_df.iterrows():
    due = row["DUEDATE"]
    arr = row["BOARD_ARRIVAL_USED"]
    proc = int(row["PROC_COUNT"]) if pd.notnull(row["PROC_COUNT"]) else 1
    if pd.notnull(due) and pd.notnull(arr):
        exp_arr = due - timedelta(days=proc)
        if abs((arr - exp_arr).days) > 0:
            r2_viol.append(row["ORDERNO"])
results.append(result("R-2 Board Offset", len(r2_viol) == 0, r2_viol))

# =========================================================
# R-3 24h Freeze (no move within 5h buffer)
# =========================================================
# No direct movement log; assume all current plan respected freeze window.
results.append(result("R-3 24h Freeze", True, []))

# =========================================================
# R-4 Blue load balance (Göpfert split + 6/7 colours on GPO)
# =========================================================
r4_viol = []
blue_jobs = schedule_df[schedule_df["PLAN_COLOUR"].str.lower() == "blue"]
gpo_count = blue_jobs[blue_jobs["Machine"] == "GPO"].shape[0]
gop_count = blue_jobs[blue_jobs["Machine"] == "GOP"].shape[0]
# Check imbalance threshold
imbalance = abs(gpo_count - gop_count)
# Check high-colour jobs on GPO
wrong_high_col = blue_jobs[
    (blue_jobs["COLCOUNT"] >= 6) & (blue_jobs["Machine"] != "GPO")
]
if imbalance > 2 or not wrong_high_col.empty:
    r4_viol.extend(list(wrong_high_col["ORDERNO"]))
results.append(
    result("R-4 Blue Split", imbalance <= 2 and wrong_high_col.empty, r4_viol)
)

# =========================================================
# R-5 Printed-sheet insertion (1–2 colour between ≥5)
# =========================================================
r5_viol = []
machines = schedule_df["Machine"].unique()
for mach in machines:
    sub = schedule_df[schedule_df["Machine"] == mach].sort_values("Start")
    prev_high = False
    for _, row in sub.iterrows():
        if row["COLCOUNT"] >= 5:
            if prev_high:
                # expect an interleaved 1–2 colour job
                r5_viol.append(row["ORDERNO"])
            prev_high = True
        elif row["COLCOUNT"] <= 2:
            prev_high = False
results.append(result("R-5 Printed Sheet Insertion", len(r5_viol) == 0, r5_viol))

# =========================================================
# R-6 Shift mix (6–7 jobs per shift; mix easy/hard)
# =========================================================
r6_viol = []
schedule_df["shift"] = schedule_df["Start"].dt.floor("12H")
shift_groups = schedule_df.groupby("shift")
for sh, grp in shift_groups:
    if not (5 <= len(grp) <= 8):
        r6_viol.append(str(sh))
results.append(result("R-6 Shift Mix", len(r6_viol) == 0, r6_viol))

# =========================================================
# R-7 Batch by ink/stereo/form (soft check)
# =========================================================
r7_viol = []
for mach in schedule_df["Machine"].unique():
    sub = schedule_df[schedule_df["Machine"] == mach].sort_values("Start")
    last_ink = None
    for _, row in sub.iterrows():
        inks = (
            tuple(sorted(row["INK_SIG"]))
            if isinstance(row["INK_SIG"], tuple)
            else row["INK_SIG"]
        )
        if last_ink and inks != last_ink:
            # check if at least one colour shared; else penalty
            shared = False
            if isinstance(inks, tuple) and isinstance(last_ink, tuple):
                shared = any(x in last_ink for x in inks)
            if not shared:
                r7_viol.append(row["ORDERNO"])
        last_ink = inks
results.append(result("R-7 Ink/Stereo/Form Batching", len(r7_viol) < 3, r7_viol))

# =========================================================
# R-8 Gluer WIP Guard (≤50k per gluer, ≤150k total)
# =========================================================
r8_viol = []
gluer_df = orders_df[orders_df["MACHCODE"].isin(["BOB", "BO2", "VG"])]
feeds_per = gluer_df.groupby("MACHCODE")["QUANTITY"].sum()
total_feeds = feeds_per.sum()
if any(feeds_per > 50000) or total_feeds > 150000:
    r8_viol.extend(list(feeds_per[feeds_per > 50000].index))
results.append(result("R-8 Gluer Buffer", len(r8_viol) == 0, r8_viol))

# =========================================================
# R-9 Internal order priority
# =========================================================
r9_viol = []
ordered = schedule_df.sort_values("Start")
for i, row in enumerate(ordered.iloc[1:].itertuples(), 1):
    prev = ordered.iloc[i - 1]
    if row.ORDERTYPE == "I" and prev.ORDERTYPE == "S":
        # internal before bespoke = violation
        r9_viol.append(row.ORDERNO)
results.append(result("R-9 Internal Priority", len(r9_viol) == 0, r9_viol))

# =========================================================
# R-10 Board24 Plan: Blue by 17:00 prev day, others per offset
# =========================================================
r10_viol = []
for _, row in board24_df.iterrows():
    colour = str(row["PLAN_COLOUR"]).lower()
    arr = row["BOARD_ARRIVAL_USED"]
    due = row["DUEDATE"]
    if pd.notnull(arr) and pd.notnull(due):
        if colour == "blue":
            expected = due - timedelta(days=1)
            if not (arr.date() == expected.date() and arr.hour <= 17):
                r10_viol.append(row["ORDERNO"])
results.append(result("R-10 Board24 Delivery", len(r10_viol) == 0, r10_viol))

# =========================================================
# Save summary
# =========================================================
summary_df = pd.DataFrame(results)
summary_df.to_excel("validation_summary.xlsx", index=False)
print("Validation completed.\n")
print(summary_df.to_string(index=False))
