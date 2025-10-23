import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
import io  # Used for download button

# =========================================================
# APP CONFIGURATION
# =========================================================
st.set_page_config(layout="wide", page_title="Job Scheduler")
st.title("🏭 Job Scheduling Optimizer")

# =========================================================
# GLOBAL CONSTANTS (from original script)
# =========================================================
TARGET_MACHINES_G1 = {"GOP", "GPO"}  # Göpfert 1
TARGET_MACHINES_G2 = {"GO2", "GP2"}  # Göpfert 2
TARGET_MACHINES = list(TARGET_MACHINES_G1 | TARGET_MACHINES_G2)


# =========================================================
# HELPER FUNCTIONS (from original script)
# =========================================================
def minute_diff(a, b):
    """Calculates difference in minutes between two datetimes."""
    return int((a - b).total_seconds() // 60)


def colour_window_minutes(colour: str):
    """Returns (low, high) minute-of-day window for a colour."""
    c = (colour or "White").strip().lower()
    if c == "blue":
        return 3 * 60, 6 * 60  # 3am - 6am
    elif c == "red":
        return 6 * 60, 8 * 60  # 6am - 8am
    else:
        return 9 * 60, 14 * 60  # 9am - 2pm


def infer_processes(row):
    """Infers process count from machine columns."""
    candidates = ["MACHINE1A", "MACHINE1B", "MACHINE2A", "MACHINE2B"]
    present = sum(
        1
        for c in candidates
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != ""
    )
    return present if present > 0 else 1


def hard_force_to_g1(machcode, colcount):
    """R-4: Re-routes high colour jobs to Göpfert 1."""
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
    """Creates a sum variable, handling empty lists."""
    if vars_list:
        s = model.NewIntVar(0, 10**9, name)
        model.Add(s == sum(vars_list))
        return s
    else:
        z = model.NewIntVar(0, 0, f"{name}_zero")
        model.Add(z == 0)
        return z


# =========================================================
# CORE SCHEDULING LOGIC
# =========================================================
def generate_schedule(df_raw, config):
    """
    Takes a raw DataFrame and config dict, returns schedule DataFrame and stats.
    """

    st.info("Starting data preprocessing...")
    df = df_raw.copy()

    # --- Data Loading & Cleaning ---
    df["DUEDATE"] = pd.to_datetime(df["DUEDATE"], errors="coerce")
    df["PLAN_COLOUR"] = df["PLAN_COLOUR"].fillna("White")
    df["MACHCODE"] = df["MACHCODE"].fillna("UNKNOWN")
    df["COLCOUNT"] = pd.to_numeric(
        df.get("COLCOUNT", pd.Series([None] * len(df))), errors="coerce"
    )

    # Filter only Göpfert jobs
    df = df[df["MACHCODE"].isin(TARGET_MACHINES)].copy()
    if df.empty:
        st.error("No valid jobs found for target machines (GOP, GPO, GO2, GP2).")
        return None, "NO_JOBS", {}

    st.info(f"Filtered down to {len(df)} Göpfert jobs.")

    # --- R-4 HARD ENFORCEMENT ---
    orig_mach = df["MACHCODE"].copy()
    df["MACHCODE"] = df.apply(
        lambda r: hard_force_to_g1(r["MACHCODE"], r["COLCOUNT"]), axis=1
    )
    remapped = (orig_mach != df["MACHCODE"]).sum()
    if remapped:
        st.info(f"R-4: Remapped {remapped} high-colour jobs to Göpfert 1.")

    # --- HORIZON SETUP ---
    horizon_start = df["DUEDATE"].min().floor("D")
    horizon_end = df["DUEDATE"].max().ceil("D") + pd.Timedelta(days=2)
    H_MIN = minute_diff(horizon_end, horizon_start)

    now = (
        pd.to_datetime(config["now_override"])
        if config["now_override"]
        else pd.Timestamp.now()
    )

    # --- DURATIONS & PRECOMPUTATIONS ---
    df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce").fillna(0)
    df["DURATION_MIN"] = (
        (df["QUANTITY"] / config["feeds_per_min"]).clip(lower=1).astype(int)
    )

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

    # --- BUILD CP MODEL ---
    st.info("Building optimization model...")
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
            gap_end = model.NewIntVar(0, H_MIN, f"gap_end_{job_id}")
            model.Add(gap_end == end + config["r5_gap_minutes"])
            gap_interval = model.NewIntervalVar(
                end, config["r5_gap_minutes"], gap_end, f"gap_{job_id}"
            )
            machine_to_intervals[machine].append(gap_interval)

        start_vars[job_id] = start
        end_vars[job_id] = end
        interval_vars[job_id] = interval

    # Machine no-overlap
    for m, ivs in machine_to_intervals.items():
        if ivs:
            model.AddNoOverlap(ivs)

    # --- OBJECTIVE ---
    total_window = safe_sum(model, window_viols, "total_window")
    total_board = safe_sum(model, board_viols, "total_board")
    total_freeze_soft = safe_sum(model, freeze_soft_devs, "total_freeze_soft")
    blue_g1_count = safe_sum(model, blue_g1_lits, "blue_g1_count")
    blue_g2_count = safe_sum(model, blue_g2_lits, "blue_g2_count")

    imbalance_diff = model.NewIntVar(-len(df), len(df), "blue_imbalance_diff")
    model.Add(imbalance_diff == blue_g1_count - blue_g2_count)
    imbalance_abs = model.NewIntVar(0, len(df), "blue_imbalance_abs")
    model.AddAbsEquality(imbalance_abs, imbalance_diff)

    if config["blue_cap"] > 0:
        model.Add(imbalance_abs <= config["blue_cap"])

    obj = model.NewIntVar(0, 10**12, "obj")
    model.Add(
        obj
        == config["w_window"] * total_window
        + config["w_board"] * total_board
        + config["w_freeze_soft"] * total_freeze_soft
        + config["w_blue_imbalance"] * imbalance_abs
    )
    model.Minimize(obj)

    # --- SOLVE ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config["time_limit"]
    solver.parameters.num_search_workers = 8

    st.info(f"Solving model... (time limit: {config['time_limit']}s)")
    status = solver.Solve(model)

    # --- OUTPUT ---
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        metrics = {
            "Total Window Violation (min)": solver.Value(total_window),
            "Total Board Violation (min)": solver.Value(total_board),
            "Total Freeze Violation (min)": solver.Value(total_freeze_soft),
            "Blue Imbalance (count)": solver.Value(imbalance_abs),
            "G1 Blue Count": solver.Value(blue_g1_count),
            "G2 Blue Count": solver.Value(blue_g2_count),
            "Objective Value": solver.Value(obj),
            "Solver Status": solver.StatusName(status),
        }

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

        out_df = pd.DataFrame(rows).sort_values(["Start", "Machine"])
        return out_df, status, metrics

    else:
        return None, status, {"Solver Status": solver.StatusName(status)}


# =========================================================
# STREAMLIT UI LAYOUT
# =========================================================

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Scheduling Parameters")
    cfg_feeds_per_min = st.number_input("Feeds per Minute", value=60, min_value=1)
    cfg_r5_gap_minutes = st.number_input(
        "R5 Colour Change Gap (min)", value=60, min_value=0
    )
    cfg_time_limit = st.number_input("Solver Time Limit (sec)", value=60, min_value=10)

    st.subheader("Objective Weights")
    cfg_w_window = st.slider("Weight: Window Violation (R-1)", 0, 10, 1)
    cfg_w_board = st.slider("Weight: Board Violation (R-2)", 0, 10, 1)
    cfg_w_freeze_soft = st.slider("Weight: Freeze Violation (R-3)", 0, 10, 5)
    cfg_w_blue_imbalance = st.slider("Weight: Blue Imbalance (R-4)", 0, 10, 2)

    st.subheader("Constraints")
    cfg_blue_cap = st.number_input(
        "Blue Imbalance Hard Cap (0 = None)", value=0, min_value=0
    )

    st.subheader("Testing")
    cfg_now_override = st.text_input("Override 'NOW' (YYYY-MM-DD HH:MM:SS)", "")

# Collect config into a dictionary
config = {
    "feeds_per_min": cfg_feeds_per_min,
    "r5_gap_minutes": cfg_r5_gap_minutes,
    "w_window": cfg_w_window,
    "w_board": cfg_w_board,
    "w_freeze_soft": cfg_w_freeze_soft,
    "w_blue_imbalance": cfg_w_blue_imbalance,
    "blue_cap": cfg_blue_cap,
    "now_override": cfg_now_override.strip() if cfg_now_override else None,
    "time_limit": cfg_time_limit,
}

# --- Main Page for File Upload and Output ---
uploaded_file = st.file_uploader(
    "Upload Job Dataset (Excel file)", type=["xlsx", "xls"]
)

if uploaded_file:
    st.success(f"File '{uploaded_file.name}' uploaded successfully.")

    if st.button("🚀 Generate Schedule", type="primary"):
        try:
            # Read the uploaded Excel file
            df_raw = pd.read_excel(uploaded_file)

            # Run the scheduling logic
            with st.spinner(
                f"Optimizing schedule... (Time limit: {config['time_limit']}s)"
            ):
                schedule_df, status, metrics = generate_schedule(df_raw, config)

            # --- Display Results ---
            if schedule_df is not None:
                st.success("✅ Solution Found!")

                st.subheader("Solution Metrics")
                st.json(metrics)

                st.subheader("Generated Schedule")
                st.dataframe(schedule_df)

                # --- Download Button ---
                @st.cache_data
                def to_excel(df):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        df.to_excel(writer, index=False, sheet_name="Schedule")
                    processed_data = output.getvalue()
                    return processed_data

                excel_data = to_excel(schedule_df)

                st.download_button(
                    label="📥 Download Schedule as Excel",
                    data=excel_data,
                    file_name="schedule_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            else:
                st.error(
                    f"❌ No feasible solution found. Status: {metrics.get('Solver Status', 'UNKNOWN')}"
                )
                if "NO_JOBS" in str(status):
                    st.warning(
                        "The uploaded file resulted in zero jobs after filtering for target machines."
                    )

        except Exception as e:
            st.exception(f"An error occurred: {e}")

else:
    st.info("Please upload your Excel dataset to begin.")
