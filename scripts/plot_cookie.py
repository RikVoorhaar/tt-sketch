# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


csv_filename = "results/cookie_2x2.csv"

df = pd.read_csv(csv_filename, index_col=0)
df.drop(columns=df.columns[0:4], inplace=True, axis=1)

for c in ["final_round_ranks"]:
    df[c] = df[c].apply(eval).apply(np.array)


def process_sl(sl):
    return np.array(
        eval(
            ",".join([s.replace("\n", "") for s in sl])
            .replace(",,", ",")
            .replace(",,", ",")
        )
    )


for c in ["true_error", "precond_error", "final_round_time"]:
    df[c] = df[c].str.split(" ").apply(process_sl)

initial_columns = [
    "rounding_method",
    "rounding_method_final",
    "max_rank",
    "run",
    "gmres_time",
]
initial_df = df[initial_columns]
final_columns = [
    "final_round_ranks",
    "true_error",
    "precond_error",
    "final_round_time",
]
final_df = df[final_columns]

row_dfs = []
for row in final_df.itertuples():
    row_idx = row[0]
    row_df = pd.DataFrame(np.stack(row[1:], axis=1), columns=final_columns)
    row_df["row_idx"] = row_idx
    row_dfs.append(row_df)

row_dfs = pd.concat(row_dfs)
initial_df.index.name = "row_idx"
plot_df = initial_df.merge(row_dfs, on="row_idx")
plot_df["total_time"] = plot_df["gmres_time"] + plot_df["final_round_time"]
# %%
for rounding_method in ["sketch", "pairwise"]:
    plot_df_rm = plot_df[plot_df["rounding_method"] == rounding_method]
    plot_df.groupby(["rounding_method", "max_rank"])["true_error"].min()
    plot_df_rm = plot_df_rm.sort_values("total_time")
    cummin = plot_df_rm["true_error"].cummin()
    plt.plot(plot_df_rm["total_time"], cummin, label=rounding_method)

plt.yscale("log")
plt.xscale("log")
plt.title("Relative error vs. time for rounding methods used in body of TT-GMRES")
plt.legend()
plt.ylabel("Relative error")
plt.xlabel("Total time")
