"""Builders for batch-upload test files (real customers from data.csv, in CSV, Excel and PDF form)."""
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Scripts"))
import churn_pipeline as cp  # noqa: E402


def customers(n=50, offset=0, keep_churn_column=True):
    """n real customers (tenure > 0) as a DataFrame in the dataset's own column layout."""
    raw = pd.read_csv(cp.DATA_PATH)
    raw = raw[raw["tenure"] > 0].iloc[offset: offset + n].reset_index(drop=True)
    return raw if keep_churn_column else raw.drop(columns=["Churn"])


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def xlsx_bytes(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def pdf_bytes(df):
    """A real PDF containing the customers as a ruled table (matplotlib draws it)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(60, 0.6 * (len(df) + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=[[str(v) for v in row] for row in df.itertuples(index=False)],
                   colLabels=list(df.columns), loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(df.columns))))
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    return buf.getvalue()


def with_changes(df, row, **changes):
    out = df.copy()
    for k, v in changes.items():
        out[k] = out[k].astype(object)
        out.loc[row, k] = v
    return out
