import numpy as np
import pandas as pd
from pathlib import Path
from utils import all_ukb_participants, build_expansion_pack, load_fid

from delphi import DAYS_PER_YEAR

# --- Load CHIP data (UKB field 30106) ---
chip_df = load_fid("30106")  # DataFrame: rows=participants, columns=array indices 0-6

# Flatten all variants into a long list
chip_variants = pd.Series(chip_df.values.ravel()).dropna().unique()
chip_variants = [v.strip() for v in chip_variants if len(str(v).strip()) > 0]

# Build tokenizer
tokenizer = {v.lower().replace(" ", "_"): i+1 for i, v in enumerate(chip_variants)}

# Build lookup map from variant string → token id
lookup = {v: tokenizer[v.lower().replace(" ", "_")] for v in chip_variants}

# --- Filter to valid UKB participants ---
ukb_subjects = all_ukb_participants()
chip_subjects = chip_df.index.to_numpy().astype(int)
is_valid = np.isin(chip_subjects, ukb_subjects)
valid_subjects = chip_subjects[is_valid]

chip_df = chip_df.loc[valid_subjects]

# --- Build token/time arrays ---
subjects = []
token_list = []
time_list = []
count_list = []

for pid, row in chip_df.iterrows():
    # Drop missing variants
    variants = [v for v in row.values if pd.notna(v)]
    if len(variants) == 0:
        continue
    tokens = [lookup[v.strip()] for v in variants]
    timesteps = [0.0] * len(tokens)  # static features → time=0
    subjects.append(pid)
    token_list.extend(tokens)
    time_list.extend(timesteps)
    count_list.append(len(tokens))

# Convert to numpy arrays
token_np = np.array(token_list, dtype=np.uint32)
time_np = np.array(time_list, dtype=np.float32)
count_np = np.array(count_list, dtype=np.uint32)
subjects = np.array(subjects, dtype=np.uint32)

# --- Build expansion pack ---
build_expansion_pack(
    token_np=token_np,
    time_np=time_np,
    count_np=count_np,
    subjects=subjects,
    tokenizer=tokenizer,
    expansion_pack="chip",
)
