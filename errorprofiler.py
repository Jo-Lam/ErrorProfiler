import json
from collections import Counter
from datetime import datetime

import jellyfish
import numpy as np
import pandas as pd


def safe_str(x):
    return "" if pd.isnull(x) else str(x)

def character_edit_positions(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0: dp[i][j] = j
            elif j == 0: dp[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    i, j = m, n
    insert_pos, delete_pos, replace_pos = [], [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and str1[i - 1] == str2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            delete_pos.append(i - 1)
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insert_pos.append(j - 1)
            j -= 1
        else:
            replace_pos.append(i - 1)
            i -= 1
            j -= 1
    return insert_pos[::-1], delete_pos[::-1], replace_pos[::-1]

def error_types_and_positions(orig, pert, variant_map=None):
    orig, pert = safe_str(orig).lower(), safe_str(pert).lower()
    etypes, positions = [], []

    if not orig and pert:
        etypes.append("Insertion")
    elif orig and not pert:
        etypes.append("Missing")
    elif orig == pert:
        etypes.append("NoError")
    else:
        # Variant
        if variant_map and orig in variant_map and variant_map[orig] == pert:
            etypes.append("Variant")
        # Edit distance logic
        ins, dele, rep = character_edit_positions(orig, pert)
        if rep:
            etypes.append("Replacement")
            positions += rep
        # Typo (single replacement, adjacent key, AND position in bounds)
        if (
            len(rep) == 1 and
            rep[0] < len(orig) and
            rep[0] < len(pert) and
            is_typo(orig[rep[0]], pert[rep[0]])
        ):
            etypes.append("Typo")
        if len(orig.split()) < len(pert.split()):
            etypes.append("TermInsertion")
        elif len(orig.split()) > len(pert.split()):
            etypes.append("TermDeletion")
        if not etypes:
            etypes.append("Edit")
    if len(etypes) > 1 and "NoError" in etypes:
        etypes = [e for e in etypes if e != "NoError"]
    return etypes, positions

def error_half_category(positions, strlen):
    if not positions: return "none"
    half = strlen / 2
    in_first = any(p < half for p in positions)
    in_second = any(p >= half for p in positions)
    if in_first and in_second:
        return "both"
    elif in_first:
        return "first_half"
    elif in_second:
        return "second_half"
    else:
        return "none"

def is_typo(a, b):
    qwerty_rows = {
        "a": "s", "b": "vn", "c": "xv", "d": "sf", "e": "wr",
        "f": "dg", "g": "fh", "h": "gj", "i": "uo", "j": "hk",
        "k": "jl", "l": "k", "m": "n", "n": "bm", "o": "ip", "p": "o",
        "q": "w", "r": "et", "s": "ad", "t": "ry", "u": "yi", "v": "cb",
        "w": "qe", "x": "zc", "y": "tu", "z": "x", "1": "2", "2": "13",
        "3": "24", "4": "35", "5": "46", "6": "57", "7": "68", "8": "79",
        "9": "80", "0": "9", "-": "_", "_": "-", "'": ""
    }
    qwerty_cols = {
        "a": "qzw", "b": "gh", "c": "df", "d": "erc", "e": "ds34",
        "f": "rvc", "g": "tbv", "h": "ybn", "i": "k89", "j": "umn",
        "k": "im", "l": "o", "m": "jk", "n": "hj", "o": "l90", "p": "0",
        "q": "a12", "r": "f45", "s": "wxz", "t": "g56", "u": "j78",
        "v": "fg", "w": "s23", "x": "sd", "y": "h67", "z": "as",
        "1": "q", "2": "qw", "3": "we", "4": "er", "5": "rt", "6": "ty",
        "7": "yu", "8": "ui", "9": "io", "0": "op", "-": "_", "_": "-", "'": ""
    }
    return (b in qwerty_rows.get(a, '')) or (b in qwerty_cols.get(a, ''))

def edit_distance_bins(ed_list):
    bins = {"1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
    for val in ed_list:
        if val == 1: bins["1"] += 1
        elif val == 2: bins["2"] += 1
        elif val == 3: bins["3"] += 1
        elif val == 4: bins["4"] += 1
        elif val >= 5: bins["5+"] += 1
    total = sum(bins.values())
    if total == 0:
        return {k: 0.0 for k in bins}
    return {k: round(v / total, 4) for k, v in bins.items()}

def dict_keys_to_str(d):
    if isinstance(d, dict):
        return {str(k): dict_keys_to_str(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [dict_keys_to_str(i) for i in d]
    elif isinstance(d, (np.generic,)):  # Converts numpy types to Python native
        return d.item()
    else:
        return d

class ErrorProfiler:
    def __init__(self, df, col_forename, col_surname, col_perturbed_forename, col_perturbed_surname, group_cols=None, rare_threshold=0.005, variant_map=None):
        self.df = df.copy()
        self.col_forename = col_forename
        self.col_surname = col_surname
        self.col_perturbed_forename = col_perturbed_forename
        self.col_perturbed_surname = col_perturbed_surname
        self.group_cols = group_cols or []
        self.rare_threshold = rare_threshold
        self.variant_map = variant_map or {}

    def compute_transformations(self):
        # For each field, calculate error types and positions
        for part, orig_col, pert_col in [
            ("forename", self.col_forename, self.col_perturbed_forename),
            ("surname", self.col_surname, self.col_perturbed_surname)
        ]:
            result = self.df.apply(
                lambda row: error_types_and_positions(
                    row[orig_col], row[pert_col], self.variant_map
                ), axis=1)
            self.df[f"{part}_etypes"] = result.apply(lambda x: x[0])
            self.df[f"{part}_err_positions"] = result.apply(lambda x: x[1])
            self.df[f"{part}_edit_distance"] = self.df.apply(
                lambda row: jellyfish.levenshtein_distance(
                    safe_str(row[orig_col]), safe_str(row[pert_col])
                ), axis=1)
            self.df[f"{part}_err_poscat"] = self.df.apply(
                lambda row: error_half_category(
                    row[f"{part}_err_positions"],
                    len(safe_str(row[orig_col]))
                ), axis=1
            )
        # Compute joint error combo
        self.df["joint_error_combo"] = self.df.apply(
            lambda row: (
                tuple(sorted(set(row["forename_etypes"]))),
                tuple(sorted(set(row["surname_etypes"])))
            ), axis=1
        )
        self.df["joint_error_combo_str"] = self.df["joint_error_combo"].apply(
            lambda t: f"forename: {','.join(t[0])} | surname: {','.join(t[1])}"
        )

    def generate_error_profile(self):
        joint_counts = self.df['joint_error_combo'].value_counts()
        total_joint = joint_counts.sum()
        joint_prob = {
            str(k): {
                "prob": round(v / total_joint, 5),
                "rare": v / total_joint < self.rare_threshold,
            } for k, v in joint_counts.items()
        }
        meta = {
            "description": (
                "Synthetic Error Profile. "
                "meta: dataset structure, joint error co-occurrence (across forename/surname), "
                "group_columns: fields for grouping (e.g., ethnicity, sex). "
                "In 'groups': error type × position joint distribution, and edit distance, per group."
            ),
            "n_total": int(len(self.df)),
            "group_columns": self.group_cols,
            "group_value_counts": {
                col: self.df[col].value_counts(normalize=True).to_dict()
                for col in self.group_cols
            },
            "joint_error_combinations": joint_prob,
            "profiler_version": "v0.4-readable",
            "generated": datetime.now().isoformat()
        }
        # Per-group
        results = []
        gb = self.df.groupby(self.group_cols) if self.group_cols else [("", self.df)]
        for key, gdf in gb:
            group = {col: val for col, val in zip(self.group_cols, key)} if self.group_cols else {}
            forename_edit = edit_distance_bins(gdf["forename_edit_distance"].tolist())
            surname_edit = edit_distance_bins(gdf["surname_edit_distance"].tolist())
            # For position: error type × position
            forename_pos_by_type = Counter()
            surname_pos_by_type = Counter()
            for _, row in gdf.iterrows():
                for et in row["forename_etypes"]:
                    forename_pos_by_type[f"{et}__{row['forename_err_poscat']}"] += 1
                for et in row["surname_etypes"]:
                    surname_pos_by_type[f"{et}__{row['surname_err_poscat']}"] += 1
            sum_forename = sum(forename_pos_by_type.values())
            sum_surname = sum(surname_pos_by_type.values())
            group.update({
                "forename_edit_distance_distribution": forename_edit,
                "surname_edit_distance_distribution": surname_edit,
                "forename_error_position_by_type": {k: round(v / sum_forename, 4) for k, v in forename_pos_by_type.items()} if sum_forename else {},
                "surname_error_position_by_type": {k: round(v / sum_surname, 4) for k, v in surname_pos_by_type.items()} if sum_surname else {},
                "n_group": int(len(gdf))
            })
            results.append(group)
        return {"meta": meta, "groups": results}

    def to_json(self, filepath=None):
        profile = self.generate_error_profile()
        profile_strkeys = dict_keys_to_str(profile)
        json_output = json.dumps(profile_strkeys, indent=2)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_output)
        return json_output

# --- Example Usage ---
# df = pd.read_csv("test_data.csv")
# profiler = ErrorProfiler(
#     df, col_forename="forename", col_surname="surname",
#     col_perturbed_forename="perturbed_forename", col_perturbed_surname="perturbed_surname",
#     group_cols=["ethgroup"]
# )
# profiler.compute_transformations()
# print(profiler.to_json())
