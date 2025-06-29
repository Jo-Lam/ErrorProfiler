import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

from corrupt.geco_corrupt import CorruptValueQuerty, position_mod_uniform


def make_types_tuple(types):
    # If empty, encode as ('NoError',) to match profile
    return tuple(sorted(types)) if types else ('NoError',)

class SyntheticGenerator:
    def __init__(
        self, 
        error_profile: Dict[str, Any], 
        N: int, 
        forename_vocab: Optional[List[str]] = None,
        surname_vocab: Optional[List[str]] = None,
        forename_alt_dict: Optional[Dict[str, List[str]]] = None,
        surname_alt_dict: Optional[Dict[str, List[str]]] = None,
        meta_cols: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
        rare_frac: float = 0.1,
        relax_rare_combinations: bool = True,
        rare_combo_prob: float = 0.2,
        keep_meta_cols: bool = True,
    ):
        self.profile = error_profile
        self.random_seed = random_seed
        self.rare_frac = rare_frac
        self._py_rng = random.Random(random_seed)
        self._np_rng = np.random.default_rng(random_seed)
        self.N = N
        self.fields = ["forename", "surname"]  # Add more fields if your profile supports it
        self.group_col = error_profile["meta"]["group_columns"][0]
        self.groups = {g[self.group_col]: g for g in error_profile["groups"]}
        self.errors = ["TermInsertion", "TermDeletion", "Replacement", "Typo", "Variant", "Inversion", "NoError"]
        self.forename_alt_dict = forename_alt_dict or {}
        self.surname_alt_dict = surname_alt_dict or {}
        self.relax_rare_combinations = relax_rare_combinations
        self.rare_combo_prob = rare_combo_prob
        self.meta_cols = meta_cols or []
        self.keep_meta_cols = keep_meta_cols
        fake = Faker()
        vocab_size = 5000

        if forename_vocab is None:
            print("[SyntheticGenerator] No forename_vocab provided. Using Faker to generate synthetic forenames.")
            self.forename_vocab = list({fake.first_name().lower() for _ in range(vocab_size)})
        else:
            self.forename_vocab = forename_vocab
        if surname_vocab is None:
            print("[SyntheticGenerator] No surname_vocab provided. Using Faker to generate synthetic surnames.")
            self.surname_vocab = list({fake.last_name().lower() for _ in range(vocab_size)})
        else:
            self.surname_vocab = surname_vocab
        
        self.vocab = {"forename": self.forename_vocab, "surname": self.surname_vocab}
        self.alt_dict = {"forename": self.forename_alt_dict, "surname": self.surname_alt_dict}

        self.qwerty_corruptor = CorruptValueQuerty(
            position_function=position_mod_uniform,
            row_prob=0.5, col_prob=0.5
        )

    def _get_group_profile(self, group_val: str) -> Dict[str, Any]:
        return self.groups.get(group_val, self.groups.get("missing", list(self.groups.values())[0]))

    def _get_position_type(self, name: str, idx: int) -> str:
        if not name or idx < 0:
            return "none"
        if idx < len(name)//2:
            return "first_half"
        else:
            return "second_half"

    def _apply_single_error(self, name: str, etype: str, rep_pairs: List[Dict[str, Any]], field: str) -> Tuple[str, int, str]:
        if etype == "Replacement":
            if not name or not rep_pairs:
                return name, -1, "none"
            total = sum(x["pct"] for x in rep_pairs)
            r = self._py_rng.uniform(0, total)
            upto = 0
            for pair in rep_pairs:
                upto += pair["pct"]
                if upto >= r:
                    from_ch, to_ch = pair["from"], pair["to"]
                    idx = name.find(from_ch)
                    if idx >= 0:
                        new_name = name[:idx] + to_ch + name[idx + 1:]
                        pos_type = self._get_position_type(name, idx)
                        return new_name, idx, pos_type
            return name, -1, "none"
        elif etype == "Typo":
            if not name:
                return name, -1, "none"
            idx = self._py_rng.randint(0, len(name)-1)
            corrupted_char = self.qwerty_corruptor.corrupt_value(name[idx])
            new_name = name[:idx] + corrupted_char + name[idx + 1:]
            pos_type = self._get_position_type(name, idx)
            return new_name, idx, pos_type
        
        elif etype == "TermInsertion":
            if not self.vocab[field]:
                return name, -1, "none"
            insert_term = self._py_rng.choice(self.vocab[field])
            if self._py_rng.uniform(0, 1) < 0.5:
                new_name = f"{insert_term} {name}"
                pos_type = "first_half"
                idx = 0
            else:
                new_name = f"{name} {insert_term}"
                pos_type = "second_half"
                idx = len(name)
            return new_name, idx, pos_type
        elif etype == "TermDeletion":
            parts = name.split()
            if len(parts) <= 1:
                return "", -1, "none"
            if self._py_rng.uniform(0, 1) < 0.5:
                new_name = " ".join(parts[1:])
                return new_name, 0, "first_half"
            else:
                new_name = " ".join(parts[:-1])
                return new_name, len(name) - len(parts[-1]), "second_half"
        elif etype == "Variant":
            variants = self.alt_dict[field].get(name.lower())
            if variants:
                var = self._py_rng.choice(variants)
                return var, 0, "both"
            return name, -1, "none"
        else:
            return name, -1, "none"  # NoError or unsupported type
        
    def _sample_joint_error_pattern(self) -> Dict[str, List[str]]:
        joint_error_combos = self.profile["meta"]["joint_error_combinations"]
        keys = list(joint_error_combos.keys())
        probs = np.array([v["prob"] for v in joint_error_combos.values()])
        rare_flags = [v.get("rare", False) for v in joint_error_combos.values()]

        while True:
            chosen = self._np_rng.choice(len(keys), p=probs/probs.sum())
            key = keys[chosen]
            is_rare = rare_flags[chosen]
            if self.relax_rare_combinations and is_rare:
                if self._py_rng.random() > self.rare_combo_prob:
                    plausible = [k for k, v in joint_error_combos.items() if not v.get("rare", False)]
                    if plausible:
                        key = self._py_rng.choice(plausible)
            errors_tuple = eval(key) if isinstance(key, str) else key
            error_dict = {f: [] for f in self.fields}
            if isinstance(errors_tuple, tuple) and len(errors_tuple) == len(self.fields):
                for i, f in enumerate(self.fields):
                    val = errors_tuple[i]
                    error_dict[f] = list(val) if isinstance(val, (list, tuple)) else [val]
            return error_dict

    def corrupt_record(self, record: Dict[str, Any], group_val: str) -> Dict[str, Any]:

        group_prof = self._get_group_profile(group_val)
        joint_errors = self._sample_joint_error_pattern()
        results = {}
        for field in self.fields:
            orig = str(record.get(field, "")).lower()
            rep_pairs = group_prof.get(f"{field}_top15_replacement_pairs", [])
            errors = joint_errors[field]
            corrupted = orig
            error_types_applied, error_positions, error_pos_types = [], [], []
            for etype in errors:
                corrupted2, idx, pos_type = self._apply_single_error(corrupted, etype, rep_pairs, field)
                if corrupted2 != corrupted:
                    error_types_applied.append(etype)
                    error_positions.append(idx)
                    error_pos_types.append(pos_type)
                corrupted = corrupted2
            results[f"orig_{field}"] = orig
            results[f"corrupted_{field}"] = corrupted
            results[f"{field}_types"] = error_types_applied
            results[f"{field}_positions"] = error_positions
            results[f"{field}_pos_types"] = error_pos_types
            results[f"edit_dist_{field}"] = sum(1 for a, b in zip(orig, corrupted) if a != b) + abs(len(orig) - len(corrupted))
        # Handle inversion as a joint error
        if any("Inversion" in results[f"{field}_types"] for field in self.fields):
            results["corrupted_forename"], results["corrupted_surname"] = results["corrupted_surname"], results["corrupted_forename"]
        results["was_error"] = int(any(results[f"corrupted_{field}"] != results[f"orig_{field}"] for field in self.fields))
        joint_combo = tuple(make_types_tuple(results[f"{field}_types"]) for field in self.fields)
        results["joint_error_combo"] = joint_combo
        return results

    def run_joint(self, df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
        group_col = group_col or self.group_col
        results = []
        for _, row in df.iterrows():
            group_val = row.get(group_col, "missing")
            meta = row.to_dict()
            for _ in range(self.N):
                rec = self.corrupt_record(meta, group_val)
                if self.keep_meta_cols:
                    # Only update with meta_cols
                    for k in self.meta_cols:
                        if k in meta:
                            rec[k] = meta[k]
                else:
                    # Keep all (legacy)
                    rec.update(meta)
                results.append(rec)
        return pd.DataFrame(results)

    def evaluate_against_profile(self, synthetic_df, thresholds=None):
        from collections import Counter

        import pandas as pd

        def compare_joint_distribution(observed_counts, expected_probs, thresholds):
            total_obs = sum(observed_counts.values())
            out = {}
            for k, expected in expected_probs.items():
                exp_prob = expected["prob"] if isinstance(expected, dict) else expected
                obs_prob = observed_counts.get(k, 0) / total_obs if total_obs > 0 else 0
                diff = obs_prob - exp_prob
                out[k] = {
                    "expected": fmt_pct(exp_prob),
                    "observed": fmt_pct(obs_prob),
                    "diff": fmt_diff(diff),
                    "flag": get_color(diff)
                }
            return out

        def evaluate_distribution(observed: dict, expected: dict, thresholds):
            total_obs = sum(observed.values())
            comparison = {}
            for k in expected:
                expected_pct = expected.get(k, 0)
                observed_pct = observed.get(k, 0) / total_obs if total_obs > 0 else 0
                diff = observed_pct - expected_pct
                comparison[k] = {
                    "expected": fmt_pct(expected_pct),
                    "observed": fmt_pct(observed_pct),
                    "diff": fmt_diff(diff),
                    "flag": get_color(diff)
                }
            return comparison

        if thresholds is None:
            thresholds = {"green": 0.05, "amber": 0.15}

        def get_color(diff):
            if abs(diff) < thresholds["green"]:
                return "green"
            elif abs(diff) < thresholds["amber"]:
                return "amber"
            else:
                return "red"

        def fmt_pct(val): return f"{val * 100:.2f}%"
        def fmt_diff(val): return f"{val * 100:+.2f}%"

        # --- Evaluate joint error co-occurrence ---
        report = {}
        matrices = {}

        group_col = self.group_col
        profile = self.profile
        group_vals = synthetic_df[group_col].unique()
        for group_val in group_vals:
            group_prof = self._get_group_profile(group_val)
            group_df = synthetic_df[synthetic_df[group_col] == group_val]
            
            # 1. Joint error co-occurrence
            obs_joint = Counter(group_df["joint_error_combo"].astype(str))
            exp_joint = profile["meta"]["joint_error_combinations"]
            joint_eval = compare_joint_distribution(obs_joint, exp_joint, thresholds)
            label = f"{group_val}:joint_error_combo"
            report[label] = joint_eval
            matrices[label] = pd.DataFrame.from_dict(joint_eval, orient="index")

            # 2. Edit distance
            for field in self.fields:
                obs_ed_bins = Counter(group_df[f"edit_dist_{field}"])
                # Bin edit distances
                ed_bins = {"1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
                for k, v in obs_ed_bins.items():
                    if k == 1: ed_bins["1"] += v
                    elif k == 2: ed_bins["2"] += v
                    elif k == 3: ed_bins["3"] += v
                    elif k == 4: ed_bins["4"] += v
                    elif k >= 5: ed_bins["5+"] += v
                exp_ed = group_prof.get(f"{field}_edit_distance_distribution", {})
                ed_eval = evaluate_distribution(ed_bins, exp_ed, thresholds)
                label_ed = f"{group_val}:{field}_edit_distance"
                report[label_ed] = ed_eval
                matrices[label_ed] = pd.DataFrame.from_dict(ed_eval, orient="index")

            # 3. Error type (flattened)
            for field in self.fields:
                obs_types = Counter(x for types in group_df[f"{field}_types"] for x in types)
                exp_types = group_prof.get(f"{field}_error_type", {})
                type_eval = evaluate_distribution(obs_types, exp_types, thresholds)
                label_type = f"{group_val}:{field}_error_type"
                report[label_type] = type_eval
                matrices[label_type] = pd.DataFrame.from_dict(type_eval, orient="index")

        return {"report": report, "matrices": matrices}
