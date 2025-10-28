import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
from datetime import timedelta, datetime
from io import BytesIO

# =========================================================
# APP CONFIGURATION
# =========================================================
st.set_page_config(layout="wide", page_title="Scheduler System")
st.title("🤖 Scheduler System")

# =========================================================
# STATIC CONFIGURATION (Business Logic)
# =========================================================
# These are less likely to change and are kept as constants.
# If you want them to be editable, they can be moved to the sidebar.
TARGET_MACHINES_G1 = {"GOP", "GPO"}  # Göpfert 1 family
TARGET_MACHINES_G2 = {"GO2", "GP2"}  # Göpfert 2 family
TARGET_MACHINES = list(TARGET_MACHINES_G1 | TARGET_MACHINES_G2)
GLUER_CODES = {"BOB", "BO2", "VG"}  # Gluer machines
GLUER_CHECK_HOURS = 6  # (narrative only, planning horizon continuous)


# =========================================================
# HELPER FUNCTIONS (Copied from your script)
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
    colc = int(colc) if pd.notna(colc) else None
    is_easy = (qty < 3000) and (colc is not None and colc <= 2)
    is_hard = (colc is not None and colc >= 5) or (qty >= 5000)
    return int(is_easy), int(is_hard)


def compute_machine_load(df, r5_gap_minutes):
    def _load(row):
        cc = row.get("COLCOUNT", None)
        gap = r5_gap_minutes if pd.notna(cc) and int(cc) >= 5 else 0
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
        s = str(val).strip().upper()
        if s.startswith("GF"):
            return s
    return None


def colour_priority(colour: str) -> int:
    """Lower is higher priority."""
    c = (colour or "").strip().lower()
    order = {"blue": 0, "red": 1, "white": 2, "green": 3, "brown": 4}
    return order.get(c, 5)


def ordertype_priority(ot: str) -> int:
    """Lower is higher priority. 'S' (bespoke/standard) before 'I' (internal)."""
    s = (ot or "").strip().upper()
    return 0 if s == "S" else 1


@st.cache_data
def to_excel(df):
    """Helper function to convert DataFrame to Excel in-memory for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Schedule")
    processed_data = output.getvalue()
    return processed_data


# =========================================================
# MAIN SCHEDULER FUNCTION
# =========================================================


def run_scheduler(
    input_file,
    # Configs from sidebar
    feeds_per_min,
    r5_gap_minutes,
    gluer_target_per,
    gluer_total_limit,
    hbuffer_days,
    now_override,
    max_solve_time,
    # Weights from sidebar
    w_window_soft,
    w_freeze_soft,
    w_blue_imbalance,
    w_r6_bal,
    w_r6_mix,
    w_setup_ink,
    w_setup_stereo,
    w_setup_form,
    w_gluer_deficit,
    w_gluer_overall_cap,
    w_r9_internal,
    blue_imbalance_hard_cap,
):
    """
    Main function to run the scheduling logic.
    Takes Streamlit inputs and returns DataFrames and status.
    """

    # Create a container for log messages
    log_container = st.expander("Show Processing Logs", expanded=True)

    def log(message):
        # This will print to the Streamlit expander
        log_container.write(message)

    try:
        # =========================================================
        # LOAD DATA
        # =========================================================
        log(f"Loading dataset from uploaded file...")
        raw = pd.read_excel(input_file)

        # Basic sanitation
        raw["DUEDATE"] = pd.to_datetime(raw["DUEDATE"], errors="coerce")
        raw["PLAN_COLOUR"] = raw["PLAN_COLOUR"].fillna("White")
        raw["MACHCODE"] = raw["MACHCODE"].fillna("UNKNOWN")
        raw["COLCOUNT"] = pd.to_numeric(
            raw.get("COLCOUNT", pd.Series([None] * len(raw))), errors="coerce"
        )
        raw["QUANTITY"] = pd.to_numeric(raw.get("QUANTITY", 0), errors="coerce").fillna(
            0
        )
        raw["ORDERTYPE"] = raw.get("ORDERTYPE", "S").fillna("S")

        # Separate gluer rows (for R-8, R-9 context), and Göpfert printable rows
        gluers_df = raw[raw["MACHCODE"].isin(GLUER_CODES)].copy()
        df = raw[raw["MACHCODE"].isin(TARGET_MACHINES)].copy()

        log(f"Filtered Göpfert jobs: {len(df)}")
        if len(df) == 0:
            st.error(
                "No valid Göpfert jobs (e.g., GOP, GPO, GO2, GP2) found in the file."
            )
            return None, None, False

        # =========================================================
        # R-4 HARD REMAP (6–7 cols → Göpfert 1)
        # =========================================================
        orig_mach = df["MACHCODE"].copy()
        df["MACHCODE"] = df.apply(
            lambda r: hard_force_to_g1(r["MACHCODE"], r["COLCOUNT"]), axis=1
        )
        remapped = (orig_mach != df["MACHCODE"]).sum()
        if remapped:
            log(f"R-4: remapped {remapped} high-colour jobs to Göpfert 1 (GOP/GPO).")

        # =========================================================
        # HORIZON SETUP
        # =========================================================
        horizon_start = df["DUEDATE"].min().floor("D") - pd.Timedelta(days=1)
        horizon_end = df["DUEDATE"].max().ceil("D") + pd.Timedelta(days=hbuffer_days)
        log(f"Horizon auto-set from {horizon_start} to {horizon_end}")
        H_MIN = minute_diff(horizon_end, horizon_start)

        try:
            now = pd.to_datetime(now_override) if now_override else pd.Timestamp.now()
            log(f"Current time ('Now') set to: {now}")
        except Exception as e:
            log(f"Invalid 'Now' override. Using current time. Error: {e}")
            now = pd.Timestamp.now()

        # =========================================================
        # DURATIONS, PROCESS COUNT, BOARD, FREEZE FLAGS + SETUP SIGS
        # =========================================================
        df["DURATION_MIN"] = (df["QUANTITY"] / feeds_per_min).clip(lower=1).astype(int)
        for c in ["MACHINE1A", "MACHINE1B", "MACHINE1C"]:
            if c not in df.columns:
                df[c] = pd.NA
        df["PROC_COUNT"] = df.apply(infer_processes, axis=1).clip(lower=1)

        # SPOCS default: board arrives (proc_count + 1) days before due @ ~17:00
        df["BOARD_ARRIVAL_USED"] = (
            df["DUEDATE"] - pd.to_timedelta(df["PROC_COUNT"] + 1, unit="D")
        ).dt.normalize() + pd.Timedelta(hours=17)

        df["PLANSTARTDATE"] = pd.to_datetime(
            df.get("PLANSTARTDATE", pd.NaT), errors="coerce"
        )
        df[["IS_EASY", "IS_HARD"]] = df.apply(
            classify_complexity, axis=1, result_type="expand"
        )

        # Setup signatures (R-7)
        df["INK_SIG"] = df.apply(build_ink_signature, axis=1)
        df["STEREO_SIG"] = df.apply(build_stereo_signature, axis=1)
        df["FORM_SIG"] = df.apply(build_form_signature, axis=1)

        # =========================================================
        # R-8: Gluer WIP summary (for mode + penalties)
        # =========================================================
        def summarize_gluer_wip(gluers: pd.DataFrame):
            summary = {}
            total = 0
            for g in sorted(GLUER_CODES):
                qty = int(gluers.loc[gluers["MACHCODE"] == g, "QUANTITY"].sum())
                summary[g] = qty
                total += qty
            return summary, total

        gluer_summary, gluer_total = summarize_gluer_wip(gluers_df)

        log("\nR-8 Gluer WIP Summary (feeds):")
        summary_data = []
        for g in sorted(gluer_summary.keys()):
            log(
                f"  {g:<3}: {gluer_summary[g]:>7,} feeds  (target ≈ {gluer_target_per:,})"
            )
            summary_data.append(
                {"Gluer": g, "Feeds": gluer_summary[g], "Target": gluer_target_per}
            )
        log(f"  TOTAL: {gluer_total:>7,} feeds  (limit ≤ {gluer_total_limit:,})")

        log_container.subheader("R-8 Gluer WIP Summary")
        log_container.dataframe(pd.DataFrame(summary_data).set_index("Gluer"))
        log_container.write(
            f"**Total Feeds:** {gluer_total:,.0f} (Limit: {gluer_total_limit:,.0f})"
        )

        # Determine if buffers are healthy (for R-9 policy)
        buffers_ok = (
            all(gluer_summary.get(g, 0) >= gluer_target_per for g in GLUER_CODES)
            and gluer_total <= gluer_total_limit
        )

        # =========================================================
        # R-5 PREPROCESSING: Pair small printed sheets
        # =========================================================
        printed_mask = (df["COLCOUNT"].fillna(0) <= 2) & (df["QUANTITY"] < 3000)
        printed_small = df[printed_mask].copy()

        batch_rows = []
        used = set()
        batch_id_seq = 1

        def family_of(m):
            return (
                "G1"
                if m in TARGET_MACHINES_G1
                else ("G2" if m in TARGET_MACHINES_G2 else "X")
            )

        df["_FAMILY"] = df["MACHCODE"].map(family_of)
        printed_small["_FAMILY"] = printed_small["MACHCODE"].map(
            family_of
        )  # for filtering

        for fam in ["G1", "G2"]:
            fam_df = printed_small[printed_small["_FAMILY"] == fam].sort_values(
                "DUEDATE"
            )
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
                    r0["DURATION_MIN"] = int(qty_sum / feeds_per_min)
                    r0["COLCOUNT"] = min(rows["COLCOUNT"].min(), 2)
                    r0["IS_EASY"], r0["IS_HARD"] = classify_complexity(r0)
                    r0["DUEDATE"] = rows["DUEDATE"].max()
                    r0["BOARD_ARRIVAL_USED"] = rows["BOARD_ARRIVAL_USED"].max()
                    r0["PLANSTARTDATE"] = pd.to_datetime(rows["PLANSTARTDATE"]).min()
                    r0["PROC_COUNT"] = int(rows["PROC_COUNT"].max())
                    # Setup sigs (uniform only)
                    r0["INK_SIG"] = (
                        rows["INK_SIG"].iloc[0]
                        if rows["INK_SIG"].nunique() == 1
                        else None
                    )
                    r0["STEREO_SIG"] = (
                        rows["STEREO_SIG"].iloc[0]
                        if rows["STEREO_SIG"].nunique() == 1
                        else None
                    )
                    r0["FORM_SIG"] = (
                        rows["FORM_SIG"].iloc[0]
                        if rows["FORM_SIG"].nunique() == 1
                        else None
                    )
                    batch_rows.append(r0)
                i = j

        if batch_rows:
            batch_df = pd.DataFrame(batch_rows).reset_index(drop=True)
            df = pd.concat([df.drop(index=list(used)), batch_df], ignore_index=True)
            log(
                f"R-5: paired {len(used)} small printed-sheet jobs into {len(batch_df)} batches."
            )

        # =========================================================
        # MACHINE LOAD PREVIEW
        # =========================================================
        load_summary = compute_machine_load(df, r5_gap_minutes)
        log("\nMachine load summary (duration + gaps):")
        log_container.subheader("Pre-Optimization Load Summary (minutes)")
        log_container.dataframe(load_summary)
        log(f"(Horizon span minutes: {H_MIN})")

        # =========================================================
        # MODEL
        # =========================================================
        log("\nBuilding optimization model...")
        model = cp_model.CpModel()

        start_vars, end_vars, interval_vars = {}, {}, {}
        freeze_soft_devs = []
        green_window_viols = []
        blue_g1_lits, blue_g2_lits = [], []
        machine_to_intervals = {m: [] for m in TARGET_MACHINES}
        machine_day_bools = {m: [] for m in TARGET_MACHINES}
        machine_night_bools = {m: [] for m in TARGET_MACHINES}
        machine_day_easy_bools = {m: [] for m in TARGET_MACHINES}
        machine_day_hard_bools = {m: [] for m in TARGET_MACHINES}
        machine_night_easy_bools = {m: [] for m in TARGET_MACHINES}
        machine_night_hard_bools = {m: [] for m in TARGET_MACHINES}
        setup_gap_terms = []
        r9_internal_terms = []

        for idx, row in df.iterrows():
            job_id = int(idx)
            machine = row["MACHCODE"]
            duration = int(row["DURATION_MIN"])
            colour = str(row["PLAN_COLOUR"]).strip().lower()
            colcount = int(row["COLCOUNT"]) if pd.notna(row["COLCOUNT"]) else None
            ordertype = str(row.get("ORDERTYPE", "S")).strip().upper()

            # decision vars
            start = model.NewIntVar(0, H_MIN, f"start_{job_id}")
            end = model.NewIntVar(0, H_MIN, f"end_{job_id}")
            interval = model.NewIntervalVar(start, duration, end, f"iv_{job_id}")
            machine_to_intervals[machine].append(interval)

            end_mod = model.NewIntVar(0, 24 * 60 - 1, f"end_mod_{job_id}")
            model.AddModuloEquality(end_mod, end, 24 * 60)

            # R-1: HARD/SOFT windows
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

            # R-2: HARD board arrival
            ba_dt = row["BOARD_ARRIVAL_USED"]
            if pd.notna(ba_dt):
                earliest_start_min = max(0, minute_diff(ba_dt, horizon_start))
                model.Add(start >= earliest_start_min)

            # R-3: freeze
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

            # R-4 blue balance
            is_blue = colour == "blue"
            eligible_blue = is_blue and (colcount is None or colcount <= 5)
            if eligible_blue:
                lit_g1 = model.NewBoolVar(f"blue_on_g1_{job_id}")
                lit_g2 = model.NewBoolVar(f"blue_on_g2_{job_id}")
                model.Add(lit_g1 == int(machine in TARGET_MACHINES_G1))
                model.Add(lit_g2 == int(machine in TARGET_MACHINES_G2))
                blue_g1_lits.append(lit_g1)
                blue_g2_lits.append(lit_g2)

            # R-5: gap
            if colcount is not None and colcount >= 5:
                gap_end = model.NewIntVar(0, H_MIN, f"gap_end_{job_id}")
                model.Add(gap_end == end + r5_gap_minutes)
                gap_interval = model.NewIntervalVar(
                    end, r5_gap_minutes, gap_end, f"gap_{job_id}"
                )
                machine_to_intervals[machine].append(gap_interval)

            # R-6 shift flags
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

            is_easy_const = int(row["IS_EASY"])
            is_hard_const = int(row["IS_HARD"])
            machine_day_bools[machine].append(is_day)
            machine_night_bools[machine].append(is_night)

            day_easy = model.NewBoolVar(f"day_easy_{job_id}")
            model.Add(day_easy == (1 if is_easy_const == 1 else 0)).OnlyEnforceIf(
                is_day
            )
            model.Add(day_easy == 0).OnlyEnforceIf(is_day.Not())
            machine_day_easy_bools[machine].append(day_easy)

            day_hard = model.NewBoolVar(f"day_hard_{job_id}")
            model.Add(day_hard == (1 if is_hard_const == 1 else 0)).OnlyEnforceIf(
                is_day
            )
            model.Add(day_hard == 0).OnlyEnforceIf(is_day.Not())
            machine_day_hard_bools[machine].append(day_hard)

            night_easy = model.NewBoolVar(f"night_easy_{job_id}")
            model.Add(night_easy == (1 if is_easy_const == 1 else 0)).OnlyEnforceIf(
                is_night
            )
            model.Add(night_easy == 0).OnlyEnforceIf(is_night.Not())
            machine_night_easy_bools[machine].append(night_easy)

            night_hard = model.NewBoolVar(f"night_hard_{job_id}")
            model.Add(night_hard == (1 if is_hard_const == 1 else 0)).OnlyEnforceIf(
                is_night
            )
            model.Add(night_hard == 0).OnlyEnforceIf(is_night.Not())
            machine_night_hard_bools[machine].append(night_hard)

            start_vars[job_id] = start
            end_vars[job_id] = end
            interval_vars[job_id] = interval

        for m, ivs in machine_to_intervals.items():
            if ivs:
                model.AddNoOverlap(ivs)

        # =========================================================
        # R-7 SOFT BATCHING
        # =========================================================
        log("Adding R-7 (soft batching) constraints...")
        job_indices = list(df.index)
        INK_SIGS = df["INK_SIG"].to_dict()
        STEREO_SIGS = df["STEREO_SIG"].to_dict()
        FORM_SIGS = df["FORM_SIG"].to_dict()
        MACH = df["MACHCODE"].to_dict()

        for a_i in range(len(job_indices)):
            i = job_indices[a_i]
            for a_j in range(a_i + 1, len(job_indices)):
                j = job_indices[a_j]
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
                    pair_w += w_setup_ink
                if same_stereo:
                    pair_w += w_setup_stereo
                if same_form:
                    pair_w += w_setup_form
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
                    scaled = model.NewIntVar(0, int(100 * H_MIN), f"scaled_gap_{i}_{j}")
                    model.Add(scaled == int(round(100 * pair_w)) * eff_gap)
                    setup_gap_terms.append(scaled)

        # =========================================================
        # R-8: gluer penalties
        # =========================================================
        gluer_pen_terms = []
        for g in GLUER_CODES:
            curr = gluer_summary.get(g, 0)
            deficit = max(0, gluer_target_per - curr)
            if deficit > 0:
                gluer_pen_terms.append(w_gluer_deficit * deficit)
        total_excess = max(0, gluer_total - gluer_total_limit)
        if total_excess > 0:
            gluer_pen_terms.append(w_gluer_overall_cap * total_excess)
        gluer_pen_total_const = sum(gluer_pen_terms)

        # =========================================================
        # R-9: internal job penalty
        # =========================================================
        if buffers_ok:
            for idx, row in df.iterrows():
                if str(row.get("ORDERTYPE", "S")).strip().upper() == "I":
                    cc = int(row["COLCOUNT"]) if pd.notna(row["COLCOUNT"]) else 0
                    dur_plus = int(row["DURATION_MIN"]) + (
                        r5_gap_minutes if cc >= 5 else 0
                    )
                    r9_internal_terms.append(w_r9_internal * dur_plus)
            r9_mode = "DELAY internal (buffers healthy)"
        else:
            r9_mode = "ALLOW internal (buffers low per R-8)"
        log(f"R-9 mode: {r9_mode}")

        # =========================================================
        # OBJECTIVE
        # =========================================================
        log("Setting objective function...")
        green_window_total = safe_sum(model, green_window_viols, "green_window_soft")
        freeze_soft_total = safe_sum(model, freeze_soft_devs, "freeze_soft_total")
        blue_g1_count = safe_sum(model, blue_g1_lits, "blue_g1_count")
        blue_g2_count = safe_sum(model, blue_g2_lits, "blue_g2_count")
        imbalance_diff = model.NewIntVar(-len(df), len(df), "blue_imbalance_diff")
        model.Add(imbalance_diff == blue_g1_count - blue_g2_count)
        imbalance_abs = model.NewIntVar(0, len(df), "blue_imbalance_abs")
        model.AddAbsEquality(imbalance_abs, imbalance_diff)
        if blue_imbalance_hard_cap is not None:
            model.Add(imbalance_abs <= blue_imbalance_hard_cap)

        # R-6
        r6_terms = []
        for m in TARGET_MACHINES:
            day_count_m = safe_sum(model, machine_day_bools[m], f"{m}_day_count")
            night_count_m = safe_sum(model, machine_night_bools[m], f"{m}_night_count")
            diff_m = model.NewIntVar(-len(df), len(df), f"{m}_day_night_diff")
            model.Add(diff_m == (day_count_m - night_count_m))
            abs_diff_m = model.NewIntVar(0, len(df), f"{m}_day_night_abs")
            model.AddAbsEquality(abs_diff_m, diff_m)
            r6_terms.append(w_r6_bal * abs_diff_m)

            # Mix penalties (Day)
            day_easy_sum = safe_sum(
                model, machine_day_easy_bools[m], f"{m}_day_easy_sum"
            )
            day_hard_sum = safe_sum(
                model, machine_day_hard_bools[m], f"{m}_day_hard_sum"
            )
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
            model.AddBoolAnd([has2_day, no_easy_day_cond]).OnlyEnforceIf(
                no_easy_day_pen
            )
            model.AddBoolOr([has2_day.Not(), no_easy_day_cond.Not()]).OnlyEnforceIf(
                no_easy_day_pen.Not()
            )
            r6_terms.append(w_r6_mix * no_easy_day_pen)
            no_hard_day_pen = model.NewBoolVar(f"{m}_no_hard_day_pen")
            model.AddBoolAnd([has2_day, no_hard_day_cond]).OnlyEnforceIf(
                no_hard_day_pen
            )
            model.AddBoolOr([has2_day.Not(), no_hard_day_cond.Not()]).OnlyEnforceIf(
                no_hard_day_pen.Not()
            )
            r6_terms.append(w_r6_mix * no_hard_day_pen)

            # Mix penalties (Night)
            night_easy_sum = safe_sum(
                model, machine_night_easy_bools[m], f"{m}_night_easy_sum"
            )
            night_hard_sum = safe_sum(
                model, machine_night_hard_bools[m], f"{m}_night_hard_sum"
            )
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
            model.AddBoolAnd([has2_night, no_easy_night_cond]).OnlyEnforceIf(
                no_easy_night_pen
            )
            model.AddBoolOr([has2_night.Not(), no_easy_night_cond.Not()]).OnlyEnforceIf(
                no_easy_night_pen.Not()
            )
            r6_terms.append(w_r6_mix * no_easy_night_pen)
            no_hard_night_pen = model.NewBoolVar(f"{m}_no_hard_night_pen")
            model.AddBoolAnd([has2_night, no_hard_night_cond]).OnlyEnforceIf(
                no_hard_night_pen
            )
            model.AddBoolOr([has2_night.Not(), no_hard_night_cond.Not()]).OnlyEnforceIf(
                no_hard_night_pen.Not()
            )
            r6_terms.append(w_r6_mix * no_hard_night_pen)

        r6_pen_sum = safe_sum(model, r6_terms, "r6_pen_sum")
        setup_gap_total = safe_sum(model, setup_gap_terms, "setup_gap_total")
        r9_internal_total_const = sum(r9_internal_terms)

        # Final objective
        obj = model.NewIntVar(0, 10**18, "obj")
        model.Add(
            obj
            == w_window_soft * green_window_total
            + w_freeze_soft * freeze_soft_total
            + w_blue_imbalance * imbalance_abs
            + r6_pen_sum
            + setup_gap_total
        )
        model.Minimize(obj)

        # =========================================================
        # SOLVE
        # =========================================================
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_solve_time
        solver.parameters.num_search_workers = (
            8  # You can also make this a sidebar option
        )

        log(f"Solving... (max time: {max_solve_time} seconds)")
        status = solver.Solve(model)

        # =========================================================
        # OUTPUT
        # =========================================================
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            status_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
            log(f"✅ Solution found (Status: {status_str})")

            solution_summary = f"""
            **Solution Summary:**
            * **R-1 (Green Window):** {solver.Value(green_window_total)}
            * **R-3 (Freeze Soft):** {solver.Value(freeze_soft_total)}
            * **R-4 (Blue Imbalance):** {solver.Value(imbalance_abs)}
            * **R-6 (Shift Pen):** {solver.Value(r6_pen_sum)}
            * **R-7 (Setup Gap):** {solver.Value(setup_gap_total)}
            * **R-8 (Gluer Pen):** {gluer_pen_total_const} (from input data)
            * **R-9 (Internal Pen):** {r9_internal_total_const}
            """
            log_container.markdown(solution_summary)

            # Build schedule dataframe
            rows = []
            for j in start_vars.keys():
                s = solver.Value(start_vars[j])
                e = solver.Value(end_vars[j])
                rows.append(
                    {
                        "ORDERNO": df.loc[j, "ORDERNO"],
                        "ORDERTYPE": df.loc[j, "ORDERTYPE"],
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
            sched = pd.DataFrame(rows).sort_values(["Start", "Machine"])
            log("Schedule DataFrame created.")

            # =====================================================
            # R-10: Board24 Load Plan generation
            # =====================================================
            def required_board_time(row):
                colour = str(row.get("PLAN_COLOUR", "")).strip().lower()
                due = row.get("DUEDATE", pd.NaT)
                if pd.isna(due):
                    return pd.NaT
                if colour == "blue":
                    return (due - pd.Timedelta(days=1)).normalize() + pd.Timedelta(
                        hours=17
                    )
                pcount = int(row.get("PROC_COUNT", 1) or 1)
                return (due - pd.Timedelta(days=pcount + 1)).normalize() + pd.Timedelta(
                    hours=17
                )

            board_df = df.copy()
            board_df["REQ_BOARD_TIME"] = board_df.apply(required_board_time, axis=1)
            board_df["COLOUR_RANK"] = board_df["PLAN_COLOUR"].map(
                lambda c: colour_priority(c)
            )
            board_df["OT_RANK"] = board_df["ORDERTYPE"].map(
                lambda x: ordertype_priority(x)
            )
            board_df["BOARD_ARRIVAL_USED"] = pd.to_datetime(
                board_df["BOARD_ARRIVAL_USED"], errors="coerce"
            )
            board_df["LATE_FLAG"] = (
                (board_df["BOARD_ARRIVAL_USED"].notna())
                & (board_df["REQ_BOARD_TIME"].notna())
                & (board_df["BOARD_ARRIVAL_USED"] > board_df["REQ_BOARD_TIME"])
            )

            # Sort for the final plan
            board_df = board_df.sort_values(
                by=["COLOUR_RANK", "OT_RANK", "REQ_BOARD_TIME", "DUEDATE"],
                ascending=[True, True, True, True],
            )
            log("R-10 Board Plan DataFrame created.")

            return sched, board_df, True

        else:
            log("❌ No feasible or optimal solution found.")
            st.error("❌ No feasible or optimal solution found within the time limit.")
            return None, None, False

    except Exception as e:
        st.error(f"An error occurred: {e}")
        log(f"Traceback: {e}")
        return None, None, False


# =========================================================
# STREAMLIT SIDEBAR (Inputs & Configuration)
# =========================================================
st.sidebar.header("1. Upload Job File")
uploaded_file = st.sidebar.file_uploader("Upload Job Excel File", type=["xlsx"])

st.sidebar.header("2. Set Parameters")

with st.sidebar.expander("Solver & Logic", expanded=True):
    max_solve_time = st.number_input(
        "Max Solve Time (seconds)", min_value=10, value=60, step=10
    )
    hbuffer_days = st.number_input(
        "Horizon Buffer (days)", min_value=0, value=4, step=1
    )
    now_override_str = st.text_input(
        "Override 'Now' (YYYY-MM-DD HH:MM)",
        value="",
        help="Leave blank to use current time",
    )

with st.sidebar.expander("Business Parameters", expanded=False):
    feeds_per_min = st.number_input("Feeds per Min", min_value=1, value=60, step=1)
    r5_gap_minutes = st.number_input(
        "R-5 Gap (mins for ≥5 colours)", min_value=0, value=60, step=5
    )
    gluer_target_per = st.number_input(
        "R-8 Gluer Target (feeds)", min_value=1, value=50000, step=1000
    )
    gluer_total_limit = st.number_input(
        "R-8 Gluer Total Limit (feeds)", min_value=1, value=150000, step=1000
    )
    blue_imbalance_hard_cap_val = st.number_input(
        "R-4 Blue Imbalance Hard Cap", min_value=1, value=100, step=1
    )  # Using 100 as a placeholder for None
    blue_imbalance_hard_cap = (
        None if blue_imbalance_hard_cap_val >= 100 else blue_imbalance_hard_cap_val
    )  # Simple way to toggle 'None'

with st.sidebar.expander("Objective Weights", expanded=False):
    st.subheader("R-1, R-3, R-4")
    w_window_soft = st.slider("W_WINDOW_SOFT (Green)", 0, 20, 1)
    w_freeze_soft = st.slider("W_FREEZE_SOFT", 0, 20, 5)
    w_blue_imbalance = st.slider("W_BLUE_IMBALANCE", 0, 20, 2)

    st.subheader("R-6")
    w_r6_bal = st.slider("W_R6_BAL (Day/Night Balance)", 0, 20, 3)
    w_r6_mix = st.slider("W_R6_MIX (Easy/Hard Mix)", 0, 20, 3)

    st.subheader("R-7 (Setup Batching)")
    w_setup_ink = st.slider("W_SETUP_INK", 0.0, 1.0, 0.05, 0.01)
    w_setup_stereo = st.slider("W_SETUP_STEREO", 0.0, 1.0, 0.07, 0.01)
    w_setup_form = st.slider("W_SETUP_FORM", 0.0, 1.0, 0.10, 0.01)

    st.subheader("R-8 (Gluer Buffers)")
    w_gluer_deficit = st.slider("W_GLUER_DEFICIT", 0, 20, 1)
    w_gluer_overall_cap = st.slider("W_GLUER_OVERALL_CAP", 0, 20, 2)

    st.subheader("R-9 (Internal Jobs)")
    w_r9_internal = st.slider("W_R9_INTERNAL", 0, 50, 20)


# =========================================================
# STREAMLIT MAIN PAGE (Button & Output)
# =========================================================
if uploaded_file is not None:
    st.info(
        "File uploaded. Adjust parameters in the sidebar and click 'Generate Schedule'."
    )

    if st.button("🚀 Generate Schedule"):
        with st.spinner(
            "Processing data and running optimization model... This may take a minute."
        ):

            # Run the main scheduler function
            sched_df, board_df, success = run_scheduler(
                input_file=uploaded_file,
                # Configs
                feeds_per_min=feeds_per_min,
                r5_gap_minutes=r5_gap_minutes,
                gluer_target_per=gluer_target_per,
                gluer_total_limit=gluer_total_limit,
                hbuffer_days=hbuffer_days,
                now_override=now_override_str,
                max_solve_time=max_solve_time,
                # Weights
                w_window_soft=w_window_soft,
                w_freeze_soft=w_freeze_soft,
                w_blue_imbalance=w_blue_imbalance,
                w_r6_bal=w_r6_bal,
                w_r6_mix=w_r6_mix,
                w_setup_ink=w_setup_ink,
                w_setup_stereo=w_setup_stereo,
                w_setup_form=w_setup_form,
                w_gluer_deficit=w_gluer_deficit,
                w_gluer_overall_cap=w_gluer_overall_cap,
                w_r9_internal=w_r9_internal,
                blue_imbalance_hard_cap=blue_imbalance_hard_cap,
            )

        if success:
            st.success("✅ Scheduling Complete!")

            # Display Schedule Output
            st.header("Generated Schedule")
            st.dataframe(sched_df)
            st.download_button(
                label="📥 Download Schedule (.xlsx)",
                data=to_excel(sched_df),
                file_name=f"schedule_output_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.ms-excel",
            )

            st.header("R-10: Board24 Load Plan")
            # Select key columns for display
            display_cols = [
                "ORDERNO",
                "PLAN_COLOUR",
                "ORDERTYPE",
                "DUEDATE",
                "REQ_BOARD_TIME",
                "BOARD_ARRIVAL_USED",
                "LATE_FLAG",
                "COLOUR_RANK",
                "OT_RANK",
                "PROC_COUNT",
            ]
            # Filter columns that exist in the dataframe
            display_cols_exist = [
                col for col in display_cols if col in board_df.columns
            ]
            st.dataframe(board_df[display_cols_exist])
            st.download_button(
                label="📥 Download Board Plan (.xlsx)",
                data=to_excel(board_df),
                file_name=f"board_plan_r10_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.ms-excel",
            )

else:
    st.info("👋 Welcome! Please upload your job Excel file using the sidebar to begin.")
