import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# =========================================================
# CONFIGURATION
# =========================================================
N_JOBS = 30
OUTPUT_PATH = "/Users/sivagar/Desktop/sherad/data/synthetic_jobs_r1to6_full.xlsx"

random.seed(42)
np.random.seed(42)

# Machine sets
G1_MACHINES = ["GPO", "GOP"]  # Göpfert 1 variants
G2_MACHINES = ["GO2", "GP2"]  # Göpfert 2 variants
ALL_MACHINES = G1_MACHINES + G2_MACHINES

COLOURS = ["Blue", "Red", "White"]

# Generate random due dates in realistic window
start_due = datetime(2025, 6, 15)
end_due = datetime(2025, 6, 30)


def random_due_date():
    delta_days = random.randint(0, (end_due - start_due).days)
    return start_due + timedelta(days=delta_days)


# =========================================================
# JOB GENERATION
# =========================================================
records = []

for i in range(N_JOBS):
    orderno = 20000 + i
    colour = random.choice(COLOURS)

    # Quantity variation (3k to 30k feeds)
    qty = random.choice([3000, 6000, 9000, 12000, 15000, 18000, 24000, 30000])

    # Colour count variation (1–7)
    colcount = random.choice([1, 2, 3, 4, 5, 6, 7])

    # Assign machine according to colour complexity (for R-4)
    if colcount >= 6:
        machcode = random.choice(G1_MACHINES)  # forced to Göpfert 1
    else:
        machcode = random.choice(ALL_MACHINES)

    # JOBSTATUSID mapping: Blue=1, Red=2, White=33
    jobstatusid = 1 if colour == "Blue" else (2 if colour == "Red" else 33)

    # Random process count (1–3)
    proc_count = random.randint(1, 3)

    duedate = random_due_date().replace(hour=0, minute=0, second=0, microsecond=0)

    # Board arrival date (R-2)
    board_arrival = duedate - timedelta(days=proc_count + 1)
    board_arrival_used = board_arrival.replace(hour=17, minute=0)

    # Duration in minutes (based on feeds/min = 60)
    duration_min = max(1, int(qty / 60))

    # Randomly create process codes
    machine1a = "PROC" if proc_count >= 1 else ""
    machine1b = "PROC" if proc_count >= 2 else ""
    machine1c = "PROC" if proc_count >= 3 else ""

    # Freeze logic (R-3): some within 5h, some 24h, some >24h
    freeze_type = random.choice(["hard", "soft", "none"])
    if freeze_type == "hard":
        planstartdate = datetime.now() + timedelta(
            hours=random.randint(1, 4)
        )  # within 5h
        freeze_flag = "Y"
    elif freeze_type == "soft":
        planstartdate = datetime.now() + timedelta(
            hours=random.randint(6, 20)
        )  # within 24h
        freeze_flag = "Y"
    else:
        planstartdate = duedate - timedelta(days=random.randint(2, 5))
        freeze_flag = "N"

    # Complexity flags (R-6)
    is_easy = int(qty < 8000 and colcount <= 3)
    is_hard = int(qty >= 12000 or colcount >= 5)

    records.append(
        {
            "ORDERNO": orderno,
            "MACHCODE": machcode,
            "JOBSTATUSID": jobstatusid,
            "PLAN_COLOUR": colour,
            "DUEDATE": duedate,
            "PLANSTARTDATE": planstartdate,
            "QUANTITY": qty,
            "COLCOUNT": colcount,
            "MACHINE1A": machine1a,
            "MACHINE1B": machine1b,
            "MACHINE1C": machine1c,
            "PROCESS_COUNT": proc_count,
            "BOARD_ARRIVAL_DATE": board_arrival,
            "DURATION_MIN": duration_min,
            "BOARD_ARRIVAL_USED": board_arrival_used,
            "FREEZE_FLAG": freeze_flag,
            "IS_EASY": is_easy,
            "IS_HARD": is_hard,
        }
    )

# =========================================================
# BUILD DATAFRAME & SAVE
# =========================================================
df = pd.DataFrame(records)
df = df.sort_values("DUEDATE").reset_index(drop=True)
df.to_excel(OUTPUT_PATH, index=False)

print(f"✅ Synthetic dataset generated: {OUTPUT_PATH}")
print(df.head(10))
