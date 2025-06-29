
# Error Profiler

**Error Profiler** is a standalone Python tool designed for secure data environments to analyse and summarise patterns in string transformation errors. It helps analysts understand how data quality issues—such as misspellings, truncations, and phonetic changes—vary by demographic or structural characteristics (e.g., age, sex, ethnicity, data source).

This tool supports validation, auditing, and fairness assessment workflows for string-matching and record linkage systems.

---

## 📌 Key Features

- **Error Extraction**: Identify transformation errors (insertions, deletions, substitutions) between original and corrupted strings.
- **Error Profiling**: Generate detailed summaries of error types, edit distances, and character positions.
- **Group-Level Analysis**: Disaggregate error patterns by user-defined categorical variables (e.g., sex, age group).
- **JSON Output**: Export profiles for use in synthetic corruption or downstream quality assessment pipelines.
- **Customisable**: Plug in your own matching/comparison functions or token parsers.

---

## 📂 Folder Structure

```
ErrorProfiler/
├──errorprofiler/
    └── errorprofiler.py         # Core profiling engine + String comparison + parsing helpers
    └── syntheticgenerator.py    # Generate Synthetic data base on error profile
    └── syntheticprofile.py      # Generate customisable error profile
    └── __init__.py
├── test_data.csv            # Example input format
├── test.py                  # Example Usage
├── README.md                # This documentation
└── output/                  # example output
    └── my_profile.json
└── alt_name_lookups
    └── forename_lookup.parquet # alternative name lookup tables from @splink
    └── surname_lookup.parquet
```

---

## 📥 Input Format

Input must be a CSV or DataFrame-like structure with at least the following columns:

| Column                  | Description                           |
|-------------------------|---------------------------------------|
| `id`                    | (Optional) Unique record identifier   | [strongly recommend]
| `forename`              | Original string                       |
| `surname`               | Original string                       |
| `forename_corrupted`    | corrupted string                      |
| `surname_corrupted`     | corrupted string                      |
| `group_vars`            | (Optional) grouping variable(s) to create the profile| 

---

## ⚙️ Usage

### 1. Load the profiler
Profile empirical name error distributions, error types, positions, edit distances, and joint error combinations for your dataset by group (e.g., by ethnicity, gender).

Typical usage:
```python
from errorprofiler import ErrorProfiler

df = pd.read_csv("test_data.csv")

profiler = ErrorProfiler(df,
                             col_forename="forename",
                             col_surname="surname",
                             col_perturbed_forename="perturbed_forename",
                             col_perturbed_surname="perturbed_surname",
                             group_cols=["ethgroup"])

profiler.compute_transformations()                             
```

### 2. Save output

```python
import json

profiler.to_json(filepath="output/my_profile.json")

```


## 📊 Output Format

The output is a nested JSON dictionary.
Each key corresponds to a unique combination of values from the specified group columns.
It also contains metadata that describes the original names.
[Future extensions: could consider extracting string features] 

--- 

### 2.5. Synthetic Error Profiles
If you do not have empirical data or wish to simulate controlled error scenarios, you can build a profile using:
```python
from errorprofiler.syntheticprofile import make_synthetic_error_profile

profile = make_synthetic_error_profile(
    N=5000,
    group_col="ethgroup",
    group_names=["White", "Black"],
    fields=["forename", "surname"],
    error_rates={
        "White": {"forename": 0.2, "surname": 0.05},
        "Black": {"forename": 0.1, "surname": 0.1}
    }
    # Optionally: error_types, error_type_dist, joint_error_probs, edit_dist_bins, edit_dist_dist
)

```
### 3. Synthetic Error Generation
`SyntheticGenerator` creates fake/corrupted string data according to any (empirical or synthetic) error profile.

## Key Features
- Separate vocabularies for forename and surname (or use Faker fallback)
- Group-specific error simulation (e.g. by ethnicity)
- Joint error pattern sampling (e.g., both names error/no error)
- Supports variants, qwerty typos, insertions/deletions, inversion, and custom types
- Metadata output (original/corrupted, error type, position, edit distance, etc.)

## Example Usage
```python
from errorprofiler.syntheticgenerator import SyntheticGenerator

gen = SyntheticGenerator(
    profile,       # From empirical data or make_synthetic_error_profile
    N=10,          # Number of synthetic draws per input row
    meta_cols=["ethgroup"],  # Columns to preserve
    keep_meta_cols=True,
    random_seed=42,
    # Optionally: forename_vocab, surname_vocab, forename_alt_dict, surname_alt_dict
)

import pandas as pd

df = pd.DataFrame({
    "ethgroup": ["White"] * 10 + ["Black"] * 10,
    # "forename" and "surname" columns are optional! Faker will be used if omitted
})

df_synth = gen.run_joint(df, group_col="ethgroup")
```
**If you do not provide forename or surname columns, names will be randomly generated via Faker.**
You can provide custom vocabularies if you want:

```python
gen = SyntheticGenerator(profile_dict, 
                        N=5, 
                        random_seed= 42, 
                        forename_vocab=forename_vocab,
                        surname_vocab = surname_vocab,
                        forename_alt_dict=alt_forename_dict,
                        surname_alt_dict=alt_surname_dict, 
                        meta_cols=["unique_id",	"ethgroup",	"gender"],
                        keep_meta_cols=True,
)
df_joint = gen.run_joint(df, group_col="ethgroup")
```
### Output Columns 
The output DataGrame includes, for each (input row x N):
| Column                  | Description                                 |
|-------------------------|---------------------------------------------|
| orig_forename           | Original forename                           |
| corrupted_forename      | Corrupted forename                          |
| forename_types          | List of error types applied to forename     |
| forename_positions      | List of positions affected in forename      |
| forename_pos_types      | List of position types (first/second half)  |
| edit_dist_forename      | Edit distance between original/corrupted    |
| orig_surname            | Original surname                            |
| corrupted_surname       | Corrupted surname                           |
| surname_types           | List of error types applied to surname      |
| surname_positions       | List of positions affected in surname       |
| surname_pos_types       | List of position types (first/second half)  |
| edit_dist_surname       | Edit distance between original/corrupted    |
| joint_error_combo       | Joint error pattern (forename, surname)     |
| ethgroup                | Group value (if input, else simulated)      |
| was_error               | Indicator if any error was applied          |
| self_defined_metadata   | Any other specified metadata (keep all by default) |
---

## ✅ Use Cases

- **Bias Auditing**: Detect if certain groups are more prone to specific types of errors.
- **Linkage Validation**: Analyze residual errors after fuzzy matching or deduplication.
- **Synthetic Model Design**: Use profiles to generate realistic, group-specific name corruptions.

---

## 🔐 Data Security Note

This tool is designed to run in secure data environments. No external calls are made. Outputs are structured for downstream automation without leaking raw strings.

---

## 📞 Contact

For issues or feature requests, contact: **[Jo LAM / Data Linkage Hub]**  
📧 **joseph.lam1@nhs.net**/**joseph.lam.18@ucl.ac.uk**
Great Ormond Street Institute of Child Health, UCL
Data Linkage Hub, NHS England

---
### Changelog
v0.2+
- Synthetic profile utility (make_synthetic_error_profile)
- New SyntheticGenerator with:
- Separate forename/surname vocab & alt_dict
- Full Faker fallback for names
- Rich metadata output
- Backward-compatible interface

### License
MIT License
