
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
├── errorprofiler.py          # Core profiling engine + String comparison + parsing helpers
├── test_data.csv             # Example input format
├── README.md                 # This documentation
└── output/
    └── my_profile.json
```

---

## 📥 Input Format

Input must be a CSV or DataFrame-like structure with at least the following columns:

| Column                  | Description                           |
|-------------------------|---------------------------------------|
| `id`                    | (Optional) Unique record identifier   |
| `forename`              | Original string                       |
| `surname`               | Original string                       |
| `perturbed_forename`    | corrupted string                      |
| `perturbed_surname`     | corrupted string                      |
| `group_vars`            | (Optional) grouping variable(s) to create the profile | 

---

## ⚙️ Usage

### 1. Load the profiler

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

---

## 📊 Output Format

The output is a nested JSON dictionary.
Each key corresponds to a unique combination of values from the specified group columns.

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
📧 **joseph.lam1@nhs.net** (NHS England)
📧 **joseph.lam.18@ucl.ac.uk** (UCL)
