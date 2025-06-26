import json

import pandas as pd

from errorprofiler import ErrorProfiler

df = pd.read_csv("test_data.csv")

profiler = ErrorProfiler(df,
                             col_forename="forename",
                             col_surname="surname",
                             col_perturbed_forename="perturbed_forename",
                             col_perturbed_surname="perturbed_surname",
                             group_cols=["ethgroup"])

profiler.compute_transformations()
profiler.to_json(filepath="my_profile.json")
