import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

OUTPUT_PATH = "/Users/sivagar/Desktop/sherad/data/synthetic_jobs_r1to6_v2_1.xlsx"
N_JOBS = 44  # a little larger to exercise R-6
MACHINES = ["GPO", "GOP", "GO2", "GP2"]
COLOURS = ["White", "Blue", "Red"]

EASY_FRAC, MEDIUM_FRAC, HARD_FRAC = 0.25, 0.40, 0.35  # slightly fewer "hard" than v2

start_date = datetime(2025, 6, 15)
end_date = datetime(2025, 7, 1)
span_days = (end_date - start_date).days

random.seed(7)
np.random.seed(7)

# caps to avoid overloading a single machine with ≥6-colour jobs
MAX_HIGH_COLOUR_PER_MACHINE = 4

per_machine_high = {m: 0 for m in MACHINES}
rows = []


def pick_machine(colcount):
    # try to distribute ≥6-colour jobs across machines, biased to G1
    if colcount >= 6:
        # prefer GOP/GPO, but respect caps
        choices = ["GPO", "GOP", "GO2", "GP2"]
        choices.sort(key=lambda m: (m not in ("GPO", "GOP"), per_machine_high[m]))
        for m in choices:
            if per_machine_high[m] < MAX_HIGH_COLOUR_PER_MACHINE:
                per_machine_high[m] += 1
                return m
        return "GPO"
    else:
        return random.choice(MACHINES)


for i in range(N_JOBS):
    orderno = 21000 + i + 1
    # complexity tier
    tier = np.random.choice(
        ["easy", "medium", "hard"], p=[EASY_FRAC, MEDIUM_FRAC, HARD_FRAC]
    )
    if tier == "easy":
        qty = np.random.randint(1200, 3000)
        colcount = np.random.randint(1, 3)
    elif tier == "medium":
        qty = np.random.randint(3200, 12000)
        colcount = np.random.randint(3, 5)
    else:
        qty = np.random.randint(12000, 26000)
        colcount = np.random.randint(5, 8)

    colour = np.random.choice(
        COLOURS, p=[0.45, 0.35, 0.20]
    )  # a bit more White/Blue than Red
    machine = pick_machine(colcount)

    # spread due dates; push big jobs away from the earliest days
    due_offset = np.random.randint(0, span_days)
    if colcount >= 6 or qty > 18000:
        due_offset = max(due_offset, 5)  # avoid cramming the first week
    duedate = start_date + timedelta(days=due_offset)

    # process count & planstart
    proc = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
    planstart = (
        duedate
        - timedelta(days=np.random.randint(2, 7))
        + timedelta(hours=np.random.randint(0, 12))
    )

    # board arrival = duedate - (proc+1..proc+3) days
    board_days_back = int(proc + np.random.randint(1, 3))  # Ensure it's a Python int
    board_date = duedate - timedelta(days=board_days_back)

    duration_min = max(1, int(qty / 60))
    is_easy = int(tier == "easy")
    is_hard = int(tier == "hard")
    freeze_flag = np.random.choice(["Y", "N"], p=[0.35, 0.65])

    rows.append(
        {
            "ORDERNO": orderno,
            "MACHCODE": machine,
            "JOBSTATUSID": np.random.choice([1, 2, 33]),
            "PLAN_COLOUR": colour,
            "DUEDATE": duedate,
            "PLANSTARTDATE": planstart,
            "QUANTITY": qty,
            "COLCOUNT": colcount,
            "MACHINE1A": "PROC" if proc >= 1 else "",
            "MACHINE1B": "PROC" if proc >= 2 else "",
            "MACHINE1C": "PROC" if proc >= 3 else "",
            "PROCESS_COUNT": proc,
            "BOARD_ARRIVAL_DATE": board_date,
            "DURATION_MIN": duration_min,
            "BOARD_ARRIVAL_USED": board_date + timedelta(hours=17),
            "FREEZE_FLAG": freeze_flag,
            "IS_EASY": is_easy,
            "IS_HARD": is_hard,
        }
    )

df = pd.DataFrame(rows)


# sanity: ensure every machine has both easy and hard somewhere (for R-6)
def ensure_mix(df):
    for m in MACHINES:
        sub = df[df["MACHCODE"] == m]
        if sub["IS_EASY"].sum() == 0:
            idx = (
                df[(df["IS_EASY"] == 1)]
                .sample(1, random_state=np.random.randint(9999))
                .index
            )
            df.loc[idx, "MACHCODE"] = m
        if sub["IS_HARD"].sum() == 0:
            idx = (
                df[(df["IS_HARD"] == 1)]
                .sample(1, random_state=np.random.randint(9999))
                .index
            )
            df.loc[idx, "MACHCODE"] = m
    return df


df = ensure_mix(df)

# export
df.to_excel(OUTPUT_PATH, index=False)
print("✅ Saved:", OUTPUT_PATH)
print(df.groupby(["MACHCODE", "IS_EASY", "IS_HARD"]).size().unstack(fill_value=0))
