import json
import pandas as pd
import numpy as np

from errorprofiler.errorprofiler import ErrorProfiler
from errorprofiler.syntheticgenerator import SyntheticGenerator
from errorprofiler.syntheticprofile import make_synthetic_error_profile


# read example data
df = pd.read_csv("test_data.csv")

# Compute error profile from actual perturbed data
profiler = ErrorProfiler(df,
                             col_forename="forename",
                             col_surname="surname",
                             col_perturbed_forename="perturbed_forename",
                             col_perturbed_surname="perturbed_surname",
                             group_cols=["ethgroup"])

profiler.compute_transformations()
profiler.to_json(filepath="output/my_profile.json")

with open("output/my_profile.json", "r", encoding="utf-8") as f:
    profile_dict = json.load(f)

# Load lookup tables - for random name replacement, and alternative name (with weights from #Splink synthetic data)
forename_lookup = pd.read_parquet('alt_name_lookups/forename_lookup.parquet')
surname_lookup = pd.read_parquet('alt_name_lookups/surname_lookup.parquet')

# Create mapping: {original_name: (alt_name_arr, alt_name_weight_arr)}
def build_altname_dict(df):
    d = {}
    for _, row in df.iterrows():
        # if these columns are lists in the parquet, you can use row['alt_name_arr'] directly
        alts = row['alt_name_arr']
        weights = row['alt_name_weight_arr']
        # but if they are strings (e.g., "[a, b, c]"), use ast.literal_eval
        if isinstance(alts, str):  # or use another test for type
            import ast
            alts = ast.literal_eval(alts)
            weights = ast.literal_eval(weights)
        d[row['original_name'].lower()] = (alts, weights)
    return d

alt_forename_dict = build_altname_dict(forename_lookup)
alt_surname_dict = build_altname_dict(surname_lookup)

forename_vocab = [str(x).lower() for x in forename_lookup['original_name'].tolist()]
surname_vocab = [str(x).lower() for x in surname_lookup['original_name'].tolist()]

gen1 = SyntheticGenerator(profile_dict, 
                        N=5, # 5 records per person
                        random_seed= 42, 
                        forename_vocab=forename_vocab,
                        surname_vocab = surname_vocab,
                        forename_alt_dict=alt_forename_dict,
                        surname_alt_dict=alt_surname_dict, 
                        meta_cols=["unique_id",	"ethgroup",	"gender"],
                        keep_meta_cols=True,
)
df_joint = gen1.run_joint(df, group_col="ethgroup")

report = gen1.evaluate_against_profile(df_joint)
matrix = report["matrices"]

# -- PART 2: Generate a synthetic error profile from a guess (no observed error data needed) --
# Assume if only have educated guess for error distribution ### 
# make synthetic error profile
profile = make_synthetic_error_profile(
    N=5000,
    group_col="ethgroup",
    group_names=["White", "Black"],
    fields=["forename", "surname"],
    error_rates={
        "White": {"forename": 0.1, "surname": 0.05},
        "Black": {"forename": 0.2, "surname": 0.1}
    },
    error_types=["Replacement", "Typo", "TermInsertion"],
    error_type_dist={
        "White": {
            "forename": {"Replacement": 0.5, "Typo": 0.3, "TermInsertion": 0.2},
            "surname": {"Replacement": 0.7, "Typo": 0.2, "TermInsertion": 0.1}
        },
        "Black": {
            "forename": {"Replacement": 0.3, "Typo": 0.6, "TermInsertion": 0.1},
            "surname": {"Replacement": 0.5, "Typo": 0.3, "TermInsertion": 0.2}
        },
    },
    joint_error_probs={
        "White": {
            "(('NoError',), ('NoError',))": 0.78,
            "(('Replacement',), ('NoError',))": 0.10,
            "(('NoError',), ('Replacement',))": 0.07,
            "(('Typo',), ('NoError',))": 0.03,
            "(('NoError',), ('Typo',))": 0.01,
            "(('TermInsertion',), ('NoError',))": 0.01,
        },
        "Black": {
            "(('NoError',), ('NoError',))": 0.7,
            "(('Replacement',), ('NoError',))": 0.13,
            "(('NoError',), ('Replacement',))": 0.11,
            "(('Typo',), ('NoError',))": 0.03,
            "(('NoError',), ('Typo',))": 0.02,
            "(('TermInsertion',), ('NoError',))": 0.01,
        },
    },
    edit_dist_bins=["1", "2", "3", "4", "5+"],
    edit_dist_dist={
        "White": {
            "forename": {"1": 0.7, "2": 0.2, "3": 0.1, "4": 0.0, "5+": 0.0},
            "surname": {"1": 0.8, "2": 0.1, "3": 0.05, "4": 0.05, "5+": 0.0},
        },
        "Black": {
            "forename": {"1": 0.4, "2": 0.3, "3": 0.2, "4": 0.05, "5+": 0.05},
            "surname": {"1": 0.5, "2": 0.2, "3": 0.2, "4": 0.1, "5+": 0.0},
        }
    },
    description="Demo: Custom profile for two ethnic groups with custom error/field structure.",
    profiler_version="v1-demo",
)


gen2 = SyntheticGenerator(
    profile,
    N=1,
    meta_cols=["ethgroup"], # or add ["gender"] if your input needs it
    keep_meta_cols=True,
    random_seed=42,
)


# Simulate group membership according to profile proportions
proportions = profile["meta"]["group_value_counts"]["ethgroup"]
n_white = int(proportions["White"] * 5000)
n_black = int(proportions["Black"] * 5000)
group = ["White"] * n_white + ["Black"] * n_black
np.random.shuffle(group)

# Simulate (minimal) input DataFrame
df2 = pd.DataFrame({
    "forename": np.random.choice(gen2.forename_vocab, size=5000),
    "surname": np.random.choice(gen2.surname_vocab, size=5000),
    "ethgroup": group,
    "unique_id": np.arange(1, 5001)
})

# Run generator
df_synth = gen2.run_joint(df, group_col="ethgroup")
df_synth[df_synth['was_error'] == 1].head(10)
