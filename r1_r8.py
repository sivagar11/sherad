import pandas as pd
from ortools.sat.python import cp_model
from datetime import timedelta

# =========================================================
# CONFIGURATION
# =========================================================
# Change if needed:
FILEPATH = "/Users/sivagar/Desktop/sherad/feasible_40jobs_r1to10_fixed.xlsx"

FEEDS_PER_MIN = 60  # feeds/min
R5_GAP_MINUTES = 60  # clean-down after ≥5 colours

# Göpfert families (printing)
TARGET_MACHINES_G1 = {"GOP", "GPO"}  # Göpfert 1 family
TARGET_MACHINES_G2 = {"GO2", "GP2"}  # Göpfert 2 family
TARGET_MACHINES = list(TARGET_MACHINES_G1 | TARGET_MACHINES_G2)

# Gluer machines (WIP buffers after print)
TARGET_GLUERS = {"BOB", "BO2", "VG"}

NOW_OVERRIDE = None  # e.g. "2025-06-18 10:00:00"

# Objective weights (tune if needed)
W_WINDOW_SOFT = 1  # used for Green only (Brown has none)
W_FREEZE_SOFT = 5
W_BLUE_IMBALANCE = 2
W_R6_BAL = 3
W_R6_MIX = 3

# R-7 batching weights (per minute of gap)
W_SETUP_INK = 0.05  # ink reuse benefit
W_SETUP_STEREO = 0.07  # stereo (plate) reuse benefit
W_SETUP_FORM = 0.10  # cutting die reuse benefit

# R-8 (Gluer WIP) weight — soft penalty on deviation
W_GLUE_WIP = 1.0

BLUE_IMBALANCE_HARD_CAP = None
HBUFFER_DAYS = 4

# R-8 targets
IDEAL_GLUE_WIP = 50_000  # per gluer
TOTAL_GLUE_MAX = 150_000  # across all gluers


# =========================================================
# HELPERS
# =========================================================
def minute_diff(a, b):
    return int((a - b).total_seconds() // 60)


def colour_window_minutes(colour: str):
    """Return (low, high) minute-of-day window for the *dispatch-ready (END)* time."""
    c = (colour or "White").strip().lower()
    if c == "blue":
        return 3 * 60, 6 * 60  # 03:00–06:00
    elif c == "red":
        return 6 * 60, 8 * 60  # 06:00–08:00
    elif c == "white":
        return 9 * 60, 14 * 60  # 09:00–14:00
    elif c == "green":
        return 6 * 60, 18 * 60  # 06:00–18:00 (SOFT)
    else:  # brown / unknown: no constraint
        return None, None


def infer_processes(row):
    candidates = ["MACHINE1A", "MACHINE1B", "MACHINE1C"]
    present = sum(
        1
        for c in candidates
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != ""
    )
    return present if present > 0 else 1


def hard_force_to_g1(machcode, colcount):
    """R-4 hard: if COLCOUNT >= 6, force to Göpfert 1 (GOP/GPO)."""
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
        s = model.NewIntVar(0, 10**12, name)
        model.Add(s == sum(vars_list))
        return s
    else:
        z = model.NewIntVar(0, 0, f"{name}_zero")
        model.Add(z == 0)
        return z


def classify_complexity(row):
    """Easy = qty < 3k and colcount ≤2; Hard = colcount ≥5 or qty ≥5k."""
    qty = float(row.get("QUANTITY", 0) or 0)
    colc = row.get("COLCOUNT", None)
    colc = int(colc) if pd.notna(colc) else 0
    is_easy = (qty < 3000) and (colc <= 2)
    is_hard = (colc >= 5) or (qty >= 5000)
    return int(is_easy), int(is_hard)


def compute_machine_load(df):
    def _load(row):
        cc = row.get("COLCOUNT", None)
        gap = 60 if pd.notna(cc) and int(cc) >= 5 else 0
        return int(row["DURATION_MIN"]) + gap

    df["_LOAD_MIN"] = df.apply(_load, axis=1)
    load = df.groupby("MACHCODE")["_LOAD_MIN"].sum().rename("minutes")
    return load


def same_family(m):
    return (
        "G1"
        if m in TARGET_MACHINES_G1
        else ("G2" if m in TARGET_MACHINES_G2 else "OTHER")
    )


def build_ink_signature(row):
    """Create a normalized tuple of non-null ink codes across COLCODE1..COLCODE7."""
    codes = []
    for k in range(1, 8):
        col = f"COLCODE{k}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
            codes.append(str(row[col]).strip())
    if not codes:
        return None
    return tuple(codes)


def build_stereo_signature(row):
    """Take first non-null STEREO* code as signature."""
    for k in range(1, 4):  # support STEREO1..STEREO3 if present
        col = f"STEREO{k}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
            return str(row[col]).strip()
    return None


def build_form_signature(row):
    """Use ORIGINATION if starts with 'GF' (Göpfert die)."""
    val = row.get("ORIGINATION", None)
    if pd.notna(val):
        s = str(val).strip()
        if s.upper().startswith("GF"):
            return s.upper()
    return None


# =========================================================
# LOAD DATA
# =========================================================
print(f"Loading dataset: {FILEPATH}")
df_all = pd.read_excel(FILEPATH)

# Basic sanitation (apply to all rows so R-8 can see gluer jobs)
df_all["DUEDATE"] = pd.to_datetime(df_all["DUEDATE"], errors="coerce")
df_all["PLAN_COLOUR"] = df_all["PLAN_COLOUR"].fillna("White")
df_all["MACHCODE"] = df_all["MACHCODE"].fillna("UNKNOWN")
df_all["COLCOUNT"] = pd.to_numeric(
    df_all.get("COLCOUNT", pd.Series([None] * len(df_all))), errors="coerce"
)
df_all["QUANTITY"] = pd.to_numeric(df_all.get("QUANTITY", 0), errors="coerce").fillna(0)

# Durations, processes etc for all rows (safe for print+gluer rows)
df_all["DURATION_MIN"] = (df_all["QUANTITY"] / FEEDS_PER_MIN).clip(lower=1).astype(int)
for c in ["MACHINE1A", "MACHINE1B", "MACHINE1C"]:
    if c not in df_all.columns:
        df_all[c] = pd.NA
df_all["PROC_COUNT"] = df_all.apply(infer_processes, axis=1).clip(lower=1)
df_all["BOARD_ARRIVAL_USED"] = (
    df_all["DUEDATE"] - pd.to_timedelta(df_all["PROC_COUNT"] + 1, unit="D")
).dt.normalize() + pd.Timedelta(hours=17)
df_all["PLANSTARTDATE"] = pd.to_datetime(
    df_all.get("PLANSTARTDATE", pd.NaT), errors="coerce"
)
df_all[["IS_EASY", "IS_HARD"]] = df_all.apply(
    classify_complexity, axis=1, result_type="expand"
)

# Setup signatures (R-7) for all rows; harmless for gluer rows
df_all["INK_SIG"] = df_all.apply(build_ink_signature, axis=1)
df_all["STEREO_SIG"] = df_all.apply(build_stereo_signature, axis=1)
df_all["FORM_SIG"] = df_all.apply(build_form_signature, axis=1)

# Create the print-only dataframe for scheduling
df = df_all[df_all["MACHCODE"].isin(TARGET_MACHINES)].copy()
print(f"Filtered Göpfert jobs: {len(df)}")

# =========================================================
# R-4 HARD REMAP (6–7 cols → Göpfert 1)
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
horizon_start = df["DUEDATE"].min().floor("D") - pd.Timedelta(days=1)
horizon_end = df["DUEDATE"].max().ceil("D") + pd.Timedelta(days=HBUFFER_DAYS)
print(f"Horizon auto-set from {horizon_start} to {horizon_end}")
H_MIN = minute_diff(horizon_end, horizon_start)
now = pd.to_datetime(NOW_OVERRIDE) if NOW_OVERRIDE else pd.Timestamp.now()

# =========================================================
# R-5 PREPROCESSING: Pair small printed sheets (<3k feeds, COLCOUNT ≤2)
#  - Prefer pairing within same machine family (G1/G2)
#  - Create a single "batched" surrogate job (sum QUANTITY, keep COLCOUNT≤2)
# =========================================================
printed_mask = (df["COLCOUNT"].fillna(0) <= 2) & (df["QUANTITY"] < 3000)
printed_small = df[printed_mask].copy()

batch_rows = []
used = set()
batch_id_seq = 1


def family_of(m):
    return (
        "G1" if m in TARGET_MACHINES_G1 else ("G2" if m in TARGET_MACHINES_G2 else "X")
    )


df["_FAMILY"] = df["MACHCODE"].map(family_of)

for fam in ["G1", "G2"]:
    fam_df = printed_small[df["_FAMILY"] == fam].sort_values("DUEDATE")
    ids = list(fam_df.index)
    i = 0
    while i < len(ids):
        if ids[i] in used:
            i += 1
            continue
        take = [ids[i]]
        qty_sum = float(df.loc[ids[i], "QUANTITY"])
        j = i + 1
        while j < len(ids) and qty_sum < 5000:
            if ids[j] not in used:
                take.append(ids[j])
                qty_sum += float(df.loc[ids[j], "QUANTITY"])
            j += 1
        if qty_sum >= 5000 and len(take) >= 2:
            used.update(take)
            rows = df.loc[take]
            order_str = "+".join(rows["ORDERNO"].astype(str).tolist())
            batch_orderno = f"BATCH{batch_id_seq}:{order_str}"
            batch_id_seq += 1
            r0 = rows.iloc[0].copy()
            r0["ORDERNO"] = batch_orderno
            r0["QUANTITY"] = qty_sum
            r0["DURATION_MIN"] = int(qty_sum / FEEDS_PER_MIN)
            r0["COLCOUNT"] = min(rows["COLCOUNT"].min(), 2)
            r0["IS_EASY"], r0["IS_HARD"] = classify_complexity(r0)
            r0["DUEDATE"] = rows["DUEDATE"].max()
            r0["BOARD_ARRIVAL_USED"] = rows["BOARD_ARRIVAL_USED"].max()
            r0["PLANSTARTDATE"] = pd.to_datetime(rows["PLANSTARTDATE"]).min()
            r0["PROC_COUNT"] = int(rows["PROC_COUNT"].max())
            # keep setup signatures if identical; else drop (None) so R-7 doesn't force them
            r0["INK_SIG"] = (
                rows["INK_SIG"].iloc[0] if (rows["INK_SIG"].nunique() == 1) else None
            )
            r0["STEREO_SIG"] = (
                rows["STEREO_SIG"].iloc[0]
                if (rows["STEREO_SIG"].nunique() == 1)
                else None
            )
            r0["FORM_SIG"] = (
                rows["FORM_SIG"].iloc[0] if (rows["FORM_SIG"].nunique() == 1) else None
            )
            batch_rows.append(r0)
        i = j

if batch_rows:
    batch_df = pd.DataFrame(batch_rows).reset_index(drop=True)
    df = pd.concat([df.drop(index=list(used)), batch_df], ignore_index=True)
    print(
        f"R-5: paired {len(used)} small printed-sheet jobs into {len(batch_df)} batches."
    )

# =========================================================
# MACHINE LOAD PREVIEW
# =========================================================
load_summary = compute_machine_load(df)
print("\nMachine load summary (duration + gaps):")
print(load_summary)
print("(Horizon span minutes:", H_MIN, ")")
print()

# =========================================================
# MODEL
# =========================================================
model = cp_model.CpModel()

start_vars, end_vars, interval_vars = {}, {}, {}
freeze_soft_devs = []
# R-1 (soft) only for Green; Brown none
green_window_viols = []

# R-4 balance (eligible blue only)
blue_g1_lits, blue_g2_lits = [], []

# R-6 containers
machine_to_intervals = {m: [] for m in TARGET_MACHINES}
machine_day_bools = {m: [] for m in TARGET_MACHINES}
machine_night_bools = {m: [] for m in TARGET_MACHINES}
machine_day_easy_bools = {m: [] for m in TARGET_MACHINES}
machine_day_hard_bools = {m: [] for m in TARGET_MACHINES}
machine_night_easy_bools = {m: [] for m in TARGET_MACHINES}
machine_night_hard_bools = {m: [] for m in TARGET_MACHINES}

for idx, row in df.iterrows():
    job_id = int(idx)
    machine = row["MACHCODE"]
    duration = int(row["DURATION_MIN"])
    colour = str(row["PLAN_COLOUR"]).strip().lower()
    colcount = int(row["COLCOUNT"]) if pd.notna(row["COLCOUNT"]) else None

    # decision vars
    start = model.NewIntVar(0, H_MIN, f"start_{job_id}")
    end = model.NewIntVar(0, H_MIN, f"end_{job_id}")
    interval = model.NewIntervalVar(start, duration, end, f"iv_{job_id}")
    machine_to_intervals[machine].append(interval)

    # minute-of-day for END (R-1 windows are about dispatch-ready)
    end_mod = model.NewIntVar(0, 24 * 60 - 1, f"end_mod_{job_id}")
    model.AddModuloEquality(end_mod, end, 24 * 60)

    # R-1: HARD windows for blue/red/white by END time; soft for green; none for brown/other
    win_low, win_high = colour_window_minutes(colour)
    if colour in {"blue", "red", "white"} and win_low is not None:
        model.Add(end_mod >= win_low)
        model.Add(end_mod <= win_high)
    elif colour == "green" and win_low is not None:
        too_early = model.NewIntVar(0, 24 * 60, f"win_early_{job_id}")
        too_late = model.NewIntVar(0, 24 * 60, f"win_late_{job_id}")
        model.Add(too_early >= win_low - end_mod)
        model.Add(too_late >= end_mod - win_high)
        win_viol = model.NewIntVar(0, 24 * 60, f"win_viol_{job_id}")
        model.AddMaxEquality(win_viol, [too_early, too_late])
        green_window_viols.append(win_viol)
    else:
        pass

    # R-2: HARD board arrival -> start cannot be before board ready
    ba_dt = row["BOARD_ARRIVAL_USED"]
    if pd.notna(ba_dt):
        earliest_start_min = max(0, minute_diff(ba_dt, horizon_start))
        model.Add(start >= earliest_start_min)

    # R-3: freeze (hard 5h, soft 24h)
    plan_start = row.get("PLANSTARTDATE", pd.NaT)
    if pd.notna(plan_start):
        planned_min = min(max(0, minute_diff(plan_start, horizon_start)), H_MIN)
        delta_to_plan_s = (plan_start - now).total_seconds()
        if 0 <= delta_to_plan_s <= 5 * 3600:
            model.Add(start == planned_min)
        elif 0 <= delta_to_plan_s <= 24 * 3600:
            dev = model.NewIntVar(0, H_MIN, f"freezeSoftDev_{job_id}")
            diff = model.NewIntVar(-H_MIN, H_MIN, f"freezeSoftDiff_{job_id}")
            model.Add(diff == start - planned_min)
            model.AddAbsEquality(dev, diff)
            freeze_soft_devs.append(dev)

    # R-4 blue balance (eligible blue: not high-colour forced)
    is_blue = colour == "blue"
    eligible_blue = is_blue and (colcount is None or colcount <= 5)
    if eligible_blue:
        lit_g1 = model.NewBoolVar(f"blue_on_g1_{job_id}")
        lit_g2 = model.NewBoolVar(f"blue_on_g2_{job_id}")
        model.Add(lit_g1 == int(machine in TARGET_MACHINES_G1))
        model.Add(lit_g2 == int(machine in TARGET_MACHINES_G2))
        blue_g1_lits.append(lit_g1)
        blue_g2_lits.append(lit_g2)

    # R-5: ≥5 colours → mandatory clean-down gap
    if colcount is not None and colcount >= 5:
        gap_end = model.NewIntVar(0, H_MIN, f"gap_end_{job_id}")
        model.Add(gap_end == end + R5_GAP_MINUTES)
        gap_interval = model.NewIntervalVar(
            end, R5_GAP_MINUTES, gap_end, f"gap_{job_id}"
        )
        machine_to_intervals[machine].append(gap_interval)

    # R-6 shift flags (by START time)
    start_mod = model.NewIntVar(0, 24 * 60 - 1, f"start_mod_{job_id}")
    model.AddModuloEquality(start_mod, start, 24 * 60)
    b_ge_6 = model.NewBoolVar(f"b_ge6_{job_id}")
    b_le_18 = model.NewBoolVar(f"b_le18_{job_id}")
    model.Add(start_mod >= 360).OnlyEnforceIf(b_ge_6)
    model.Add(start_mod < 360).OnlyEnforceIf(b_ge_6.Not())
    model.Add(start_mod <= 1079).OnlyEnforceIf(b_le_18)
    model.Add(start_mod > 1079).OnlyEnforceIf(b_le_18.Not())

    is_day = model.NewBoolVar(f"is_day_{job_id}")
    model.AddBoolAnd([b_ge_6, b_le_18]).OnlyEnforceIf(is_day)
    model.AddBoolOr([b_ge_6.Not(), b_le_18.Not()]).OnlyEnforceIf(is_day.Not())
    is_night = model.NewBoolVar(f"is_night_{job_id}")
    model.Add(is_night + is_day == 1)

    # complexity constants
    is_easy_const = int(row["IS_EASY"])
    is_hard_const = int(row["IS_HARD"])

    machine_day_bools[machine].append(is_day)
    machine_night_bools[machine].append(is_night)

    day_easy = model.NewBoolVar(f"day_easy_{job_id}")
    model.Add(day_easy == (1 if is_easy_const == 1 else 0)).OnlyEnforceIf(is_day)
    model.Add(day_easy == 0).OnlyEnforceIf(is_day.Not())
    machine_day_easy_bools[machine].append(day_easy)

    day_hard = model.NewBoolVar(f"day_hard_{job_id}")
    model.Add(day_hard == (1 if is_hard_const == 1 else 0)).OnlyEnforceIf(is_day)
    model.Add(day_hard == 0).OnlyEnforceIf(is_day.Not())
    machine_day_hard_bools[machine].append(day_hard)

    night_easy = model.NewBoolVar(f"night_easy_{job_id}")
    model.Add(night_easy == (1 if is_easy_const == 1 else 0)).OnlyEnforceIf(is_night)
    model.Add(night_easy == 0).OnlyEnforceIf(is_night.Not())
    machine_night_easy_bools[machine].append(night_easy)

    night_hard = model.NewBoolVar(f"night_hard_{job_id}")
    model.Add(night_hard == (1 if is_hard_const == 1 else 0)).OnlyEnforceIf(is_night)
    model.Add(night_hard == 0).OnlyEnforceIf(is_night.Not())
    machine_night_hard_bools[machine].append(night_hard)

    start_vars[job_id] = start
    end_vars[job_id] = end
    interval_vars[job_id] = interval

# Machine no-overlap
for m, ivs in machine_to_intervals.items():
    if ivs:
        model.AddNoOverlap(ivs)

# =========================================================
# R-7 SOFT BATCHING: minimize gaps between same-group jobs on the same machine
# =========================================================
setup_gap_terms = []
job_indices = list(df.index)

INK_SIGS = df["INK_SIG"].to_dict()
STEREO_SIGS = df["STEREO_SIG"].to_dict()
FORM_SIGS = df["FORM_SIG"].to_dict()
MACH = df["MACHCODE"].to_dict()

for ai in range(len(job_indices)):
    i = job_indices[ai]
    for aj in range(ai + 1, len(job_indices)):
        j = job_indices[aj]
        if MACH[i] != MACH[j]:
            continue

        same_ink = (INK_SIGS.get(i) is not None) and (
            INK_SIGS.get(i) == INK_SIGS.get(j)
        )
        same_stereo = (STEREO_SIGS.get(i) is not None) and (
            STEREO_SIGS.get(i) == STEREO_SIGS.get(j)
        )
        same_form = (FORM_SIGS.get(i) is not None) and (
            FORM_SIGS.get(i) == FORM_SIGS.get(j)
        )

        pair_w = 0.0
        if same_ink:
            pair_w += W_SETUP_INK
        if same_stereo:
            pair_w += W_SETUP_STEREO
        if same_form:
            pair_w += W_SETUP_FORM

        if pair_w <= 0.0:
            continue

        o_ij = model.NewBoolVar(f"o_{i}_before_{j}")
        o_ji = model.NewBoolVar(f"o_{j}_before_{i}")
        model.Add(o_ij + o_ji == 1)

        gap_ij = model.NewIntVar(0, H_MIN, f"gap_{i}_to_{j}")
        gap_ji = model.NewIntVar(0, H_MIN, f"gap_{j}_to_{i}")

        model.Add(start_vars[j] - end_vars[i] >= 0).OnlyEnforceIf(o_ij)
        model.Add(gap_ij == start_vars[j] - end_vars[i]).OnlyEnforceIf(o_ij)
        model.Add(gap_ij == 0).OnlyEnforceIf(o_ij.Not())

        model.Add(start_vars[i] - end_vars[j] >= 0).OnlyEnforceIf(o_ji)
        model.Add(gap_ji == start_vars[i] - end_vars[j]).OnlyEnforceIf(o_ji)
        model.Add(gap_ji == 0).OnlyEnforceIf(o_ji.Not())

        eff_gap = model.NewIntVar(0, H_MIN, f"eff_gap_{i}_{j}")
        model.Add(eff_gap == gap_ij + gap_ji)

        if pair_w == int(pair_w):
            setup_gap_terms.append(int(pair_w) * eff_gap)
        else:
            scaled = model.NewIntVar(0, 100 * H_MIN, f"scaled_gap_{i}_{j}")
            model.Add(scaled == int(round(100 * pair_w)) * eff_gap)
            setup_gap_terms.append(scaled)

# =========================================================
# R-8: GLUER WIP GUARD (soft)
#   - Aim ~50k feeds per gluer (|WIP - 50k| penalty)
#   - Total WIP across gluers ≤ 150k (overage penalty)
#   - Uses df_all (not filtered) to see live WIP rows
# =========================================================
gluer_df = df_all[df_all["MACHCODE"].isin(TARGET_GLUERS)].copy()
# Filter out canceled/completed if present: ORDERSTATUS "X" or "C"
if "ORDERSTATUS" in gluer_df.columns:
    gluer_df = gluer_df[~gluer_df["ORDERSTATUS"].isin(["X", "C"])]

glue_dev_terms = []
if not gluer_df.empty:
    gluer_wip = gluer_df.groupby("MACHCODE")["QUANTITY"].sum().to_dict()
    total_wip_val = int(sum(gluer_wip.values()))
    print("\nR-8 Gluer WIP Summary (feeds):")
    for g, v in gluer_wip.items():
        print(f"  {g:<4}: {int(v):>8,} feeds  (target ≈ 50,000)")
    print(f"  TOTAL: {int(total_wip_val):>8,} feeds  (limit ≤ 150,000)")

    # Per-gluer deviation |WIP - 50k|
    for g, v in gluer_wip.items():
        dev = model.NewIntVar(0, 10**9, f"glue_dev_{g}")
        # v is a constant; AbsEq is fine using constant on RHS
        model.AddAbsEquality(dev, int(v) - IDEAL_GLUE_WIP)
        glue_dev_terms.append(dev)

    glue_dev_sum = safe_sum(model, glue_dev_terms, "glue_dev_sum")

    # Total overage max(total - 150k, 0)
    total_wip = model.NewIntVar(0, 10**9, "glue_total_wip")
    model.Add(total_wip == sum(int(v) for v in gluer_wip.values()))
    over_total = model.NewIntVar(0, 10**9, "glue_over_total")
    # over_total = max(0, total_wip - TOTAL_GLUE_MAX)
    # CP-SAT trick: over_total >= total_wip - TOTAL_GLUE_MAX and over_total >= 0 ; minimize it
    tmp = model.NewIntVar(-(10**9), 10**9, "glue_tmp_diff")
    model.Add(tmp == total_wip - TOTAL_GLUE_MAX)
    # over_total = max(tmp, 0)
    # Implement with two constraints and objective minimization via equality with Max?
    # Use AddMaxEquality:
    model.AddMaxEquality(over_total, [tmp, model.NewConstant(0)])

    # Combine R-8 terms; scale if W_GLUE_WIP not integer
    if W_GLUE_WIP == int(W_GLUE_WIP):
        glue_pen_total = model.NewIntVar(0, 10**12, "glue_pen_total")
        model.Add(glue_pen_total == int(W_GLUE_WIP) * (glue_dev_sum + over_total))
    else:
        glue_pen_total = model.NewIntVar(0, 10**12, "glue_pen_total_scaled")
        model.Add(
            glue_pen_total == int(round(100 * W_GLUE_WIP)) * (glue_dev_sum + over_total)
        )
else:
    print("\nR-8: No gluer jobs present (no WIP penalty).")
    glue_pen_total = model.NewIntVar(0, 0, "glue_pen_total_zero")
    model.Add(glue_pen_total == 0)

# =========================================================
# OBJECTIVE (R-6 + blue balance + freeze soft + green window soft + R-7 + R-8)
# =========================================================
green_window_total = safe_sum(model, green_window_viols, "green_window_soft")
freeze_soft_total = safe_sum(model, freeze_soft_devs, "freeze_soft_total")

blue_g1_count = safe_sum(model, blue_g1_lits, "blue_g1_count")
blue_g2_count = safe_sum(model, blue_g2_lits, "blue_g2_count")
imbalance_diff = model.NewIntVar(-len(df), len(df), "blue_imbalance_diff")
model.Add(imbalance_diff == blue_g1_count - blue_g2_count)
imbalance_abs = model.NewIntVar(0, len(df), "blue_imbalance_abs")
model.AddAbsEquality(imbalance_abs, imbalance_diff)
if BLUE_IMBALANCE_HARD_CAP is not None:
    model.Add(imbalance_abs <= BLUE_IMBALANCE_HARD_CAP)

# R-6 per machine: day/night balance AND mix penalties
r6_terms = []
for m in TARGET_MACHINES:
    day_count_m = safe_sum(model, machine_day_bools[m], f"{m}_day_count")
    night_count_m = safe_sum(model, machine_night_bools[m], f"{m}_night_count")

    # |day - night|
    diff_m = model.NewIntVar(-len(df), len(df), f"{m}_day_night_diff")
    model.Add(diff_m == (day_count_m - night_count_m))
    abs_diff_m = model.NewIntVar(0, len(df), f"{m}_day_night_abs")
    model.AddAbsEquality(abs_diff_m, diff_m)
    r6_terms.append(W_R6_BAL * abs_diff_m)

    # Mix penalties if that shift has at least 2 jobs
    # --- Day shift ---
    day_easy_sum = safe_sum(model, machine_day_easy_bools[m], f"{m}_day_easy_sum")
    day_hard_sum = safe_sum(model, machine_day_hard_bools[m], f"{m}_day_hard_sum")

    has2_day = model.NewBoolVar(f"{m}_has2_day")
    model.Add(day_count_m >= 2).OnlyEnforceIf(has2_day)
    model.Add(day_count_m <= 1).OnlyEnforceIf(has2_day.Not())

    no_easy_day_cond = model.NewBoolVar(f"{m}_no_easy_day_cond")
    model.Add(day_easy_sum == 0).OnlyEnforceIf(no_easy_day_cond)
    model.Add(day_easy_sum != 0).OnlyEnforceIf(no_easy_day_cond.Not())

    no_hard_day_cond = model.NewBoolVar(f"{m}_no_hard_day_cond")
    model.Add(day_hard_sum == 0).OnlyEnforceIf(no_hard_day_cond)
    model.Add(day_hard_sum != 0).OnlyEnforceIf(no_hard_day_cond.Not())

    no_easy_day_pen = model.NewBoolVar(f"{m}_no_easy_day_pen")
    model.AddBoolAnd([has2_day, no_easy_day_cond]).OnlyEnforceIf(no_easy_day_pen)
    model.AddBoolOr([has2_day.Not(), no_easy_day_cond.Not()]).OnlyEnforceIf(
        no_easy_day_pen.Not()
    )
    r6_terms.append(W_R6_MIX * no_easy_day_pen)

    no_hard_day_pen = model.NewBoolVar(f"{m}_no_hard_day_pen")
    model.AddBoolAnd([has2_day, no_hard_day_cond]).OnlyEnforceIf(no_hard_day_pen)
    model.AddBoolOr([has2_day.Not(), no_hard_day_cond.Not()]).OnlyEnforceIf(
        no_hard_day_pen.Not()
    )
    r6_terms.append(W_R6_MIX * no_hard_day_pen)

    # --- Night shift ---
    night_easy_sum = safe_sum(model, machine_night_easy_bools[m], f"{m}_night_easy_sum")
    night_hard_sum = safe_sum(model, machine_night_hard_bools[m], f"{m}_night_hard_sum")

    has2_night = model.NewBoolVar(f"{m}_has2_night")
    model.Add(night_count_m >= 2).OnlyEnforceIf(has2_night)
    model.Add(night_count_m <= 1).OnlyEnforceIf(has2_night.Not())

    no_easy_night_cond = model.NewBoolVar(f"{m}_no_easy_night_cond")
    model.Add(night_easy_sum == 0).OnlyEnforceIf(no_easy_night_cond)
    model.Add(night_easy_sum != 0).OnlyEnforceIf(no_easy_night_cond.Not())

    no_hard_night_cond = model.NewBoolVar(f"{m}_no_hard_night_cond")
    model.Add(night_hard_sum == 0).OnlyEnforceIf(no_hard_night_cond)
    model.Add(night_hard_sum != 0).OnlyEnforceIf(no_hard_night_cond.Not())

    no_easy_night_pen = model.NewBoolVar(f"{m}_no_easy_night_pen")
    model.AddBoolAnd([has2_night, no_easy_night_cond]).OnlyEnforceIf(no_easy_night_pen)
    model.AddBoolOr([has2_night.Not(), no_easy_night_cond.Not()]).OnlyEnforceIf(
        no_easy_night_pen.Not()
    )
    r6_terms.append(W_R6_MIX * no_easy_night_pen)

    no_hard_night_pen = model.NewBoolVar(f"{m}_no_hard_night_pen")
    model.AddBoolAnd([has2_night, no_hard_night_cond]).OnlyEnforceIf(no_hard_night_pen)
    model.AddBoolOr([has2_night.Not(), no_hard_night_cond.Not()]).OnlyEnforceIf(
        no_hard_night_pen.Not()
    )
    r6_terms.append(W_R6_MIX * no_hard_night_pen)

r6_pen_sum = safe_sum(model, r6_terms, "r6_pen_sum")

# R-7 total setup gap penalty
setup_gap_total = safe_sum(model, setup_gap_terms, "setup_gap_total")

# ----- Final objective -----
obj_terms = [
    W_WINDOW_SOFT * green_window_total,
    W_FREEZE_SOFT * freeze_soft_total,
    W_BLUE_IMBALANCE * imbalance_abs,
    r6_pen_sum,
    setup_gap_total,
    glue_pen_total,  # R-8 soft penalty
]
obj = model.NewIntVar(0, 10**15, "obj")
model.Add(obj == sum(obj_terms))
model.Minimize(obj)

# =========================================================
# SOLVE
# =========================================================
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60
solver.parameters.num_search_workers = 8

print("Solving...")
status = solver.Solve(model)

# =========================================================
# OUTPUT
# =========================================================
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(
        f"✅ Solution: green_win={solver.Value(green_window_total)} "
        f"freeze_soft={solver.Value(freeze_soft_total)} "
        f"blue_imbalance={solver.Value(imbalance_abs)} "
        f"r6_pen={solver.Value(r6_pen_sum)} "
        f"setup_gap={solver.Value(setup_gap_total)} "
        f"gluer_pen={solver.Value(glue_pen_total)}"
    )

    rows = []
    for j in start_vars.keys():
        s = solver.Value(start_vars[j])
        e = solver.Value(end_vars[j])
        rows.append(
            {
                "ORDERNO": df.loc[j, "ORDERNO"],
                "Machine": df.loc[j, "MACHCODE"],
                "PLAN_COLOUR": df.loc[j, "PLAN_COLOUR"],
                "COLCOUNT": df.loc[j, "COLCOUNT"],
                "Start": horizon_start + timedelta(minutes=s),
                "End": horizon_start + timedelta(minutes=e),
                "Duration_min": df.loc[j, "DURATION_MIN"],
                "DUEDATE": df.loc[j, "DUEDATE"],
                "BOARD_ARRIVAL_USED": df.loc[j, "BOARD_ARRIVAL_USED"],
                "PROC_COUNT": df.loc[j, "PROC_COUNT"],
                "IS_EASY": df.loc[j, "IS_EASY"],
                "IS_HARD": df.loc[j, "IS_HARD"],
                "INK_SIG": df.loc[j, "INK_SIG"],
                "STEREO_SIG": df.loc[j, "STEREO_SIG"],
                "FORM_SIG": df.loc[j, "FORM_SIG"],
            }
        )
    out = pd.DataFrame(rows).sort_values(["Start", "Machine"])
    out.to_excel("schedule_output_r1to8_final.xlsx", index=False)
    print("📁 Schedule saved to schedule_output_r1to8_final.xlsx")
else:
    print("❌ No feasible solution found. Status:", solver.StatusName(status))
