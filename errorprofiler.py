import json
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations

import jellyfish
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Utility Functions
# -----------------------------

def safe_str(x):
    """Ensure string or empty string for string functions."""
    return "" if pd.isnull(x) else str(x)

def character_replacement_details(str1, str2):
    replacements = []
    for i, (a, b) in enumerate(zip(str1, str2)):
        if a != b:
            replacements.append((a, b, i))
    return replacements

def get_noerror_rate_by_group(self, field="forename_transformation_type"):
    """
    Returns a dictionary of no-error rates for each group, based on the current DataFrame.
    field: "forename_transformation_type" or "surname_transformation_type"
    Returns: {group_key: p_noerror}
    """
    noerror_map = {}
    group_cols = self.group_cols
    groupby_obj = self.df.groupby(group_cols) if group_cols else [((), self.df)]
    for key, gdf in groupby_obj:
        # Flexible: key can be tuple or scalar depending on group_cols
        # Check for no-error (empty list or only contains "NoError")
        n = len(gdf)
        n_noerr = gdf[field].apply(lambda l: (not l) or (l == ["NoError"])).sum()
        noerror_map[key if isinstance(key, tuple) else (key,)] = n_noerr / n if n else 0.0
    return noerror_map

def inject_noerror(self, profile_dict, noerror_map=None, field="forename_error_type", default_noerror=0.15):
    """
    Patch a generated profile_dict (from generate_error_profile) to inject a "NoError" probability for each group.
    Supports any set of group columns.
    Args:
        profile_dict: output of generate_error_profile()
        noerror_map: dict mapping group keys (tuple of group values, or string for single column) to probability.
        field: which error_type field to patch ("forename_error_type" or "surname_error_type")
        default_noerror: fallback if group not found in map
    Returns:
        Patched profile_dict (in-place).
    """
    group_cols = profile_dict["meta"].get("group_columns", [])

    for group in profile_dict["groups"]:
        # Build the key (tuple or str)
        if len(group_cols) == 0:
            group_key = ()
        elif len(group_cols) == 1:
            group_key = (group.get(group_cols[0], "missing") or "missing").lower()
        else:
            group_key = tuple((group.get(col, "missing") or "missing").lower() for col in group_cols)

        # Look up, flexibly, for tuple or single
        p_noerror = None
        if noerror_map:
            if group_key in noerror_map:
                p_noerror = noerror_map[group_key]
            elif isinstance(group_key, tuple) and len(group_key) == 1 and group_key[0] in noerror_map:
                p_noerror = noerror_map[group_key[0]]
        if p_noerror is None:
            p_noerror = default_noerror

        etypes = group.get(field, {})
        total = sum(etypes.values())
        scale = 1 - p_noerror
        if total > 0:
            for k in etypes:
                etypes[k] = etypes[k] * scale / total
        etypes["NoError"] = p_noerror
        group[field] = etypes

    return profile_dict


def edit_distance_bins(ed_list):
    # Accepts a list of integers
    bins = {"1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
    for val in ed_list:
        if val == 1:
            bins["1"] += 1
        elif val == 2:
            bins["2"] += 1
        elif val == 3:
            bins["3"] += 1
        elif val == 4:
            bins["4"] += 1
        elif val >= 5:
            bins["5+"] += 1
    total = sum(bins.values())
    if total == 0:
        return {k: 0.0 for k in bins}
    return {k: v / total for k, v in bins.items()}


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

"""def is_phonetic_match(a, b):
    a, b = safe_str(a), safe_str(b)
    if not a or not b: return False
    try:
        return jellyfish.soundex(a) == jellyfish.soundex(b)
    except Exception:
        return False"""

def term_insertion_deletion(orig, perturbed):
    orig_terms = safe_str(orig).lower().split()
    pert_terms = safe_str(perturbed).lower().split()
    inserted = [(t, i+1) for i, t in enumerate(pert_terms) if t not in orig_terms]
    deleted = [(t, i+1) for i, t in enumerate(orig_terms) if t not in pert_terms]
    return inserted, deleted

def detect_variant(orig, pert, variant_map):
    o = safe_str(orig).lower()
    p = safe_str(pert).lower()
    if o == p or not o or not p:
        return False
    # Check both directions in mapping
    return variant_map.get(o, None) == p or variant_map.get(p, None) == o

def detect_inversion(forename, surname, pert_forename, pert_surname):
    f, s = safe_str(forename).lower(), safe_str(surname).lower()
    pf, ps = safe_str(pert_forename).lower(), safe_str(pert_surname).lower()
    return (f in ps and s in pf) and (f or s)

def add_cooccurrence(profile, error_types, field_prefix):
    # e.g. error_types = ["Typo", "Phonetic"]
    if len(error_types) > 1:
        for pair in combinations(sorted(set(error_types)), 2):
            key = "__".join(pair)
            profile[field_prefix + "_error_cooccurrence"][key] += 1

def character_edit_positions(str1, str2):
    # Returns (insertions, deletions, replacements) as position lists
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

# --- Helper for position categorisation ---
def error_half_category(positions, strlen):
    """Categorize error positions as 'first_half', 'second_half', or 'both'."""
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

def group_key_to_str(key):
    if isinstance(key, tuple):
        return "|".join(str(k) for k in key)
    return str(key)

def dict_keys_to_str(d):
    """Recursively convert all dict keys to strings for JSON serialization."""
    if isinstance(d, dict):
        return {str(k): dict_keys_to_str(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [dict_keys_to_str(i) for i in d]
    else:
        return d

def is_missing(orig, pert):
    return bool(safe_str(orig)) and not bool(safe_str(pert))
# -----------------------------
# Configurable Variant Map
# -----------------------------
DEFAULT_VARIANT_MAP = {
    "liz": "elizabeth", "beth": "elizabeth", "betty": "elizabeth",
    "bob": "robert", "bobby": "robert",
    "jim": "james", "jimmy": "james",
    "kate": "katherine", "kathy": "katherine",
    "tom": "thomas", "tommy": "thomas",
    "anne": "ann", "john": "jon"
    # extend as needed...
}

# -----------------------------
# Profiler Class
# -----------------------------

class ErrorProfiler:
    def __init__(self, df, col_forename, col_surname, col_perturbed_forename, col_perturbed_surname, group_cols=None, variant_map=None):
        self.df = df.copy()
        self.col_forename = col_forename
        self.col_surname = col_surname
        self.col_perturbed_forename = col_perturbed_forename
        self.col_perturbed_surname = col_perturbed_surname
        self.group_cols = group_cols or []
        self.variant_map = variant_map or DEFAULT_VARIANT_MAP


    def compute_transformations(self):
        for col in [self.col_forename, self.col_surname, self.col_perturbed_forename, self.col_perturbed_surname]:
            self.df[col] = self.df[col].astype(str).str.lower()    
        for part, orig_col, pert_col in [
            ('forename', self.col_forename, self.col_perturbed_forename),
            ('surname', self.col_surname, self.col_perturbed_surname)
        ]:
            # Detect inversion and missing first
            if part == "forename":
                self.df['inversion'] = self.df.apply(
                    lambda row: detect_inversion(
                        row[self.col_forename], row[self.col_surname],
                        row[self.col_perturbed_forename], row[self.col_perturbed_surname]
                    ), axis=1
                )
            self.df[f'{part}_is_missing'] = self.df.apply(
                lambda row: is_missing(row[orig_col], row[pert_col]), axis=1
            )

            # Apply term insertion/deletion logic only if not missing or inverted
            def smart_term_change(row):
                if row[f'{part}_is_missing'] or (part == "forename" and row.get('inversion', False)):
                    return [], []
                return term_insertion_deletion(row[orig_col], row[pert_col])

            self.df[f'{part}_inserted_terms'], self.df[f'{part}_deleted_terms'] = zip(*self.df.apply(
                smart_term_change, axis=1
            ))

            self.df[f'{part}_is_variant'] = self.df.apply(
                lambda row: detect_variant(row[orig_col], row[pert_col], self.variant_map), axis=1
            )
            self.df[f'{part}_replacements'] = self.df.apply(
                lambda row: character_replacement_details(safe_str(row[orig_col]), safe_str(row[pert_col])), axis=1
            )
            self.df[f'{part}_has_typo'] = self.df[f'{part}_replacements'].apply(
                lambda replacements: any(is_typo(a, b) for a, b, _ in replacements)
            )
            self.df[f'{part}_edit_distance'] = self.df.apply(
                lambda row: jellyfish.levenshtein_distance(safe_str(row[orig_col]), safe_str(row[pert_col]))
                if safe_str(row[orig_col]) and safe_str(row[pert_col]) else 0, axis=1
            )
            # Character-level insert/delete positions
            self.df[[f'{part}_insert_positions', f'{part}_delete_positions', f'{part}_replace_positions']] = self.df.apply(
                lambda row: pd.Series(character_edit_positions(
                    safe_str(row[orig_col]), safe_str(row[pert_col])
                )), axis=1
            )
            strlen = self.df[orig_col].apply(lambda x: len(safe_str(x)))
            self.df[f'{part}_replace_half_category'] = [
                error_half_category(poslist, l) for poslist, l in zip(self.df[f'{part}_replace_positions'], strlen)
            ]
            self.df[f'{part}_insert_half_category'] = [
                error_half_category(poslist, l) for poslist, l in zip(self.df[f'{part}_insert_positions'], strlen)
            ]
            self.df[f'{part}_delete_half_category'] = [
                error_half_category(poslist, l) for poslist, l in zip(self.df[f'{part}_delete_positions'], strlen)
            ]

            # Build error type list with suppression if missing/inversion
            def get_types(row):
                types = []
                if row[f'{part}_is_missing']:
                    types.append("Missing")
                elif part == "forename" and row.get('inversion', False):
                    types.append("Inversion")
                else:
                    if row[f'{part}_inserted_terms']:
                        types.append("TermInsertion")
                    if row[f'{part}_deleted_terms']:
                        types.append("TermDeletion")
                if row[f'{part}_is_variant']:
                    types.append("Variant")
                if row[f'{part}_has_typo']:
                    types.append("Typo")
                if row[f'{part}_replacements']:
                    types.append("Replacement")
                return types

            self.df[f'{part}_transformation_type'] = self.df.apply(get_types, axis=1)

        # Combine all error labels across fields
        self.df['error_type_label'] = self.df.apply(
            lambda row: list(set(
                row['forename_transformation_type'] +
                row['surname_transformation_type']
            )), axis=1
        )


    def generate_error_profile(self, mode = "all"):
        """
        mode: "all" = denominator is all rows (default)
        "error" = denominator is only rows with any error in that field
        """
        profile_denoms_forename = defaultdict(int)   # group_key -> denominator
        profile_denoms_surname = defaultdict(int)

        profile = defaultdict(lambda: defaultdict(lambda: Counter()))
        edit_distance_dist = defaultdict(lambda: defaultdict(list))
        example_rows = {}
        replacement_pair_counter = defaultdict(lambda: defaultdict(Counter))

        # ADD these for the new summaries:
        char_pos_summary = defaultdict(lambda: defaultdict(Counter))
        term_name_summary = defaultdict(lambda: defaultdict(list))

        # Your group columns (e.g., ['race', 'sex'])
        group_cols = self.group_cols

        # Pass 1: Collect all group stats
        group_n = defaultdict(int)
        for idx, row in self.df.iterrows():
            key = tuple(row[col] for col in self.group_cols)
            error_in_forename = bool(row['forename_transformation_type'])
            error_in_surname = bool(row['surname_transformation_type'])
            if not error_in_forename:
                profile[key]['forename_error_type']['NoError'] += 1
            # Count "no error" surname
            if not error_in_surname:
                profile[key]['surname_error_type']['NoError'] += 1
            if mode == "all":
                profile_denoms_forename[key] += 1
                profile_denoms_surname[key] += 1
            elif mode == "error":
                profile_denoms_forename[key] += int(error_in_forename)
                profile_denoms_surname[key] += int(error_in_surname)
            group_n[key] += 1
            if key not in example_rows:
                example_rows[key] = {
                    'forename': row[self.col_forename],
                    'perturbed_forename': row[self.col_perturbed_forename],
                    'surname': row[self.col_surname],
                    'perturbed_surname': row[self.col_perturbed_surname]
                }
            # Error type frequency
            for t in row['forename_transformation_type']:
                profile[key]['forename_error_type'][t] += 1
            for t in row['surname_transformation_type']:
                profile[key]['surname_error_type'][t] += 1
            for label in row['error_type_label']:
                profile[key]['error_category'][label] += 1
            add_cooccurrence(profile[key], row['forename_transformation_type'], 'forename')
            add_cooccurrence(profile[key], row['surname_transformation_type'], 'surname')
            edit_distance_dist[key]['forename'].append(row['forename_edit_distance'])
            edit_distance_dist[key]['surname'].append(row['surname_edit_distance'])
            # Half categories (proportion)
            char_pos_summary[key]['forename_replace_half'][row['forename_replace_half_category']] += 1
            char_pos_summary[key]['forename_insert_half'][row['forename_insert_half_category']] += 1
            char_pos_summary[key]['forename_delete_half'][row['forename_delete_half_category']] += 1
            char_pos_summary[key]['surname_replace_half'][row['surname_replace_half_category']] += 1
            char_pos_summary[key]['surname_insert_half'][row['surname_insert_half_category']] += 1
            char_pos_summary[key]['surname_delete_half'][row['surname_delete_half_category']] += 1
            # Letter replacement pairs
            for a, b, i in row['forename_replacements']:
                replacement_pair_counter[key]['forename'][(a, b)] += 1
            for a, b, i in row['surname_replacements']:
                replacement_pair_counter[key]['surname'][(a, b)] += 1

            # Term-level summary - do not include in profile, but can be used for internal analysis.
            # term_name_summary[key]['forename_inserted_term_names'].extend(row['forename_inserted_term_names'])
            # term_name_summary[key]['forename_deleted_term_names'].extend(row['forename_deleted_term_names'])
            # term_name_summary[key]['surname_inserted_term_names'].extend(row['surname_inserted_term_names'])
            # term_name_summary[key]['surname_deleted_term_names'].extend(row['surname_deleted_term_names'])
        # Pass 2: Format as "groups" list of dicts
        groups_list = []
        for key, stats in profile.items():
            group_dict = {col: val for col, val in zip(group_cols, key)}
            group_dict["n"] = group_n[key]

            forename_pairs = replacement_pair_counter[key]['forename']
            surname_pairs = replacement_pair_counter[key]['surname']
            forename_total = sum(forename_pairs.values())
            surname_total = sum(surname_pairs.values())
            forename_top15 = forename_pairs.most_common(15)
            surname_top15 = surname_pairs.most_common(15)

            forename_error_counts = stats['forename_error_type']
            surname_error_counts = stats['surname_error_type']
            forename_total_n = profile_denoms_forename[key]
            surname_total_n = profile_denoms_surname[key]
            # Ensure "NoError" key always present, even if 0
            if "NoError" not in forename_error_counts:
                forename_error_counts["NoError"] = 0
            if "NoError" not in surname_error_counts:
                surname_error_counts["NoError"] = 0

            def normalize_dict_erroraware(counts, denom):
                return {k: v / denom if denom else 0.0 for k, v in counts.items()}
            
            def normalize_half_counter(counter):
                keys = ["first_half", "second_half", "both", "none"]
                total = sum(counter.get(k, 0) for k in keys)
                if total == 0:
                    return {k: 0.0 for k in keys}
                return {k: counter.get(k, 0) / total for k in keys}

            group_dict.update({
                "forename_error_type": normalize_dict_erroraware(forename_error_counts, forename_total_n),
                "surname_error_type": normalize_dict_erroraware(surname_error_counts, surname_total_n),
                "error_category": self._normalize_dict(stats['error_category']),
                "forename_error_cooccurrence": self._normalize_dict(stats['forename_error_cooccurrence']),
                "surname_error_cooccurrence": self._normalize_dict(stats['surname_error_cooccurrence']),
                "forename_edit_distance_distribution": edit_distance_bins(edit_distance_dist[key]['forename']),
                "surname_edit_distance_distribution": edit_distance_bins(edit_distance_dist[key]['surname']),
                "forename_replace_half": normalize_half_counter(char_pos_summary[key]['forename_replace_half']),
                "forename_insert_half": normalize_half_counter(char_pos_summary[key]['forename_insert_half']),
                "forename_delete_half": normalize_half_counter(char_pos_summary[key]['forename_delete_half']),
                "surname_replace_half": normalize_half_counter(char_pos_summary[key]['surname_replace_half']),
                "surname_insert_half": normalize_half_counter(char_pos_summary[key]['surname_insert_half']),
                "surname_delete_half": normalize_half_counter(char_pos_summary[key]['surname_delete_half']),
                "forename_top15_replacement_pairs": [
                    {"from": a, "to": b, "pct": (n / forename_total) if forename_total > 0 else 0.0}
                    for (a, b), n in forename_top15
                ],
                "surname_top15_replacement_pairs": [
                    {"from": a, "to": b, "pct": (n / surname_total) if surname_total > 0 else 0.0}
                    for (a, b), n in surname_top15
                ],
            })
            groups_list.append(group_dict)

        noerror_rate_by_group_forename = {}
        noerror_rate_by_group_surname = {}
        for group_dict in groups_list:
            # Build group key tuple, consistent with group_cols
            key = tuple(group_dict[col] for col in group_cols)
            skey = group_key_to_str(key)
            noerror_rate_by_group_forename[key] = group_dict['forename_error_type'].get('NoError', 0.0)
            noerror_rate_by_group_surname[key] = group_dict['surname_error_type'].get('NoError', 0.0)

        nowstr = datetime.now().isoformat()
        meta = {
            "n_total": len(self.df),
            "noerror_rate_by_group_forename": noerror_rate_by_group_forename,
            "noerror_rate_by_group_surname": noerror_rate_by_group_surname,
            "group_columns": self.group_cols,
            "group_value_counts": {
                col: self.df[col].value_counts(normalize=True).to_dict()
                for col in self.group_cols
            },
            "profiler_version": "version 0.1",
            "Mode": mode,
            "generated": nowstr,
        }            

        # add meta  
        return {"groups": groups_list, "meta": meta }

    def _normalize_dict(self, d):
        total = sum(d.values())
        return {k: v / total for k, v in d.items()} if total > 0 else {}

    def to_json(self, filepath=None, **kwargs):
        """
        Save the profile as JSON to the specified filepath if given.
        Returns the JSON string as well.
        Additional kwargs are passed to json.dumps (e.g. indent).
        """
        profile = self.generate_error_profile()
        profile_strkeys = dict_keys_to_str(profile)    # <-- ADD THIS LINE
        json_output = json.dumps(profile_strkeys, indent=2, **kwargs)  # <-- Serialize this one!
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_output)
        return json_output


    def plot_summary(self, group_key=None, keys_to_plot=None):
        """
        Plot all distribution/histogram keys in the JSON profile.
        Optionally, specify a single group_key (as a string), or plot all groups.
        Optionally, specify which keys (list) to plot; else, all dict/list keys will be plotted.
        """
        profile = self.generate_error_profile()
        groups = [group_key] if group_key else list(profile.keys())
        for key in groups:
            print(f"\n==== Plotting group: {key} ====")
            d = profile[key]
            # By default, plot all dicts/lists except meta
            plot_keys = keys_to_plot if keys_to_plot else [
                k for k in d.keys() if (isinstance(d[k], dict) or isinstance(d[k], list)) and k != "meta"
            ]
            for k in plot_keys:
                data = d[k]
                plt.figure(figsize=(6, 4))
                if isinstance(data, dict):
                    # Remove empty/no-value keys
                    data = {kk: vv for kk, vv in data.items() if vv}
                    if not data: continue
                    pd.Series(data).sort_values(ascending=False).plot(kind='bar')
                    plt.ylabel("Probability" if all(isinstance(v, float) and v <= 1 for v in data.values()) else "Count")
                    plt.title(f"{k} ({key})")
                    plt.tight_layout()
                    plt.show()
                elif isinstance(data, list):
                    # Only plot if list has data
                    if not data: continue
                    pd.Series(data).value_counts().sort_values(ascending=False).plot(kind='bar')
                    plt.ylabel("Count")
                    plt.title(f"{k} ({key})")
                    plt.tight_layout()
                    plt.show()
                else:
                    # If it's a number or unsupported, skip
                    continue

"""
class ErrorProfilerHigherConditional(ErrorProfiler):
    def generate_error_profile(self):
        profile = defaultdict(lambda: defaultdict(lambda: Counter()))
        edit_distance_dist = defaultdict(lambda: defaultdict(list))
        error_type_sets = defaultdict(Counter)
        example_rows = {}
        meta_per_group = {}

        for idx, row in self.df.iterrows():
            key = tuple(row[col] for col in self.group_cols)
            if key not in example_rows:
                example_rows[key] = {
                    'forename': row[self.col_forename],
                    'perturbed_forename': row[self.col_perturbed_forename],
                    'surname': row[self.col_surname],
                    'perturbed_surname': row[self.col_perturbed_surname]
                }
            # Per-field and category counts
            for t in row['forename_transformation_type']:
                profile[key]['forename_error_type'][t] += 1
            for t in row['surname_transformation_type']:
                profile[key]['surname_error_type'][t] += 1
            for label in row['error_type_label']:
                profile[key]['error_category'][label] += 1
            add_cooccurrence(profile[key], row['forename_transformation_type'], 'forename')
            add_cooccurrence(profile[key], row['surname_transformation_type'], 'surname')
            edit_distance_dist[key]['forename'].append(row['forename_edit_distance'])
            edit_distance_dist[key]['surname'].append(row['surname_edit_distance'])
            # Track higher-order: the full set of errors (frozenset, so order doesn't matter)
            error_type_sets[key][frozenset(row['error_type_label'])] += 1

        # Build export
        nowstr = datetime.now().isoformat()
        final_profile = {}
        for key, stats in profile.items():
            # Convert set counts to string for JSON (frozenset is not JSON-serializable)
            set_counter = error_type_sets[key]
            set_counter_str = {", ".join(sorted(s)): c for s, c in set_counter.items()}
            total = sum(set_counter.values())
            set_counter_norm = {k: v / total for k, v in set_counter_str.items()} if total > 0 else {}
            final_profile[str(key)] = {
                "forename_error_type": self._normalize_dict(stats['forename_error_type']),
                "surname_error_type": self._normalize_dict(stats['surname_error_type']),
                "error_category": self._normalize_dict(stats['error_category']),
                "forename_error_cooccurrence": self._normalize_dict(stats['forename_error_cooccurrence']),
                "surname_error_cooccurrence": self._normalize_dict(stats['surname_error_cooccurrence']),
                "forename_edit_distance": edit_distance_dist[key]['forename'],
                "surname_edit_distance": edit_distance_dist[key]['surname'],
                "error_type_set_distribution": set_counter_norm,
                "meta": {
                    "example": example_rows.get(key, {}),
                    "n": len(edit_distance_dist[key]['forename']),
                    "profiler_version": "2.0-highcond",
                    "generated": nowstr
                }
            }
        return final_profile

    def to_json(self, filepath=None):
        profile = self.generate_error_profile()
        json_output = json.dumps(profile, indent=2)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_output)
        return json_output
"""
# -----------------------------
# Example Usage
# -----------------------------

if __name__ == "__main__":
    data = {
        "forename": ["liz", "anne", "jim", "tom", "maria"],
        "surname": ["smith", "lee", "park", "o'reilly", "gonzalez"],
        "perturbed_forename": ["elizabeth", "anne john", "jimmy", "tim", "maria elena"],
        "perturbed_surname": ["smith", "lee", "park", "oreilly", "gonzalez"],
        "race": ["Asian", "White", "Asian", "Black", "Other"],
        "sex": ["F", "F", "M", "M", "F"]
    }
    df = pd.DataFrame(data)
    profiler = ErrorProfiler(df,
                             col_forename="forename",
                             col_surname="surname",
                             col_perturbed_forename="perturbed_forename",
                             col_perturbed_surname="perturbed_surname",
                             group_cols=["race", "sex"])
    profiler.compute_transformations()
    print(profiler.to_json())
#   profiler.plot_summary(keys_to_plot=['forename_replace_half', 'forename_inserted_term_names'])  # Only certain keys

"""    profiler = ErrorProfilerHigherConditional(
        df,
        col_forename="forename",
        col_surname="surname",
        col_perturbed_forename="perturbed_forename",
        col_perturbed_surname="perturbed_surname",
        group_cols=["race", "sex"]
    )
    profiler.compute_transformations()
    print(profiler.to_json())
    profiler.plot_summary()"""




