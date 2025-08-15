# Copyright (c) 2025 takotime808

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, anderson_ksamp, zscore, entropy
from scipy.spatial.distance import euclidean, minkowski, mahalanobis
from scipy.linalg import inv
import matplotlib.pyplot as plt
from io import StringIO, BytesIO

st.set_page_config(page_title="Two Sample Comparison App", layout="wide")
st.title("📊 Compare Two Uploaded Samples")

# === File Uploads ===
st.sidebar.header("Step 1: Upload Files")
uploaded_file1 = st.sidebar.file_uploader("Upload CSV for Sample 1", type=["csv"])
uploaded_file2 = st.sidebar.file_uploader("Upload CSV for Sample 2", type=["csv"])


def get_filterable_columns(df):
    return df.select_dtypes(include=[np.number, "category", "object"]).columns.tolist()


def filter_dataframe(df, label):
    st.subheader(f"🔍 Filter: {label}")
    filtered_df = df.copy()
    with st.expander(f"Filter options for {label}", expanded=False):
        for col in get_filterable_columns(df):
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                selected_range = st.slider(f"{label}: {col}", min_val, max_val, (min_val, max_val))
                filtered_df = filtered_df[(df[col] >= selected_range[0]) & (df[col] <= selected_range[1])]
            else:
                options = df[col].dropna().unique().tolist()
                selected = st.multiselect(f"{label}: {col}", options, default=options)
                filtered_df = filtered_df[df[col].isin(selected)]
    return filtered_df


def detect_sampling_method(df):
    """Try to infer the sampling method used to collect the data.

    The function uses simple heuristics based on the presence of grouping
    columns or constant step sizes in numeric columns.
    It returns a tuple of (method, justification).
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        diffs = df[col].dropna().diff().dropna()
        if len(diffs) > 0 and diffs.nunique() == 1 and diffs.iloc[0] != 0:
            step = diffs.iloc[0]
            return (
                "Systematic Sampling",
                f"Values in column '{col}' increment by a constant step of {step}",
            )

    for col in df.columns:
        low = col.lower()
        if any(key in low for key in ["strata", "stratum", "group", "cluster"]):
            counts = df[col].value_counts()
            if not counts.empty:
                imbalance = (counts.max() - counts.min()) / counts.mean()
                if imbalance < 0.5:
                    return (
                        "Stratified Sampling",
                        f"Column '{col}' has relatively balanced group sizes",
                    )
                return (
                    "Cluster Sampling",
                    f"Column '{col}' indicates clusters with varying sizes",
                )

    return (
        "Simple Random Sampling",
        "No constant increments or grouping columns detected",
    )


if uploaded_file1 and uploaded_file2:
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)

    method1, just1 = detect_sampling_method(df1)
    method2, just2 = detect_sampling_method(df2)

    with st.expander("Detected Sampling Methods", expanded=False):
        st.markdown(f"**Sample 1:** {method1}  \n*Justification:* {just1}")
        st.markdown(f"**Sample 2:** {method2}  \n*Justification:* {just2}")

    # === Optional Filters ===
    df1 = filter_dataframe(df1, "Sample 1")
    df2 = filter_dataframe(df2, "Sample 2")

    st.sidebar.header("Step 2: Select Columns")
    col1 = st.sidebar.selectbox("Column from Sample 1", df1.select_dtypes(include=np.number).columns)
    col2 = st.sidebar.selectbox("Column from Sample 2", df2.select_dtypes(include=np.number).columns)

    data1 = df1[col1].dropna()
    data2 = df2[col2].dropna()

    # === Transform Options ===
    st.sidebar.header("Step 3: Preprocessing")
    log_transform = st.sidebar.checkbox("Log transform")
    log_base = st.sidebar.selectbox("Log base", ["e", "10"], disabled=not log_transform)
    standardize = st.sidebar.checkbox("Standardize (z-score)")

    def preprocess(data, label):
        transformed = data.copy()
        if log_transform:
            if (transformed <= 0).any():
                st.warning(f"{label}: Log transform skipped due to non-positive values.")
            else:
                transformed = np.log(transformed) if log_base == "e" else np.log10(transformed)
        if standardize:
            transformed = zscore(transformed)
        return transformed

    data1 = preprocess(data1, "Sample 1")
    data2 = preprocess(data2, "Sample 2")

    # === Metric Selection ===
    st.sidebar.header("Step 4: Select Metrics")
    with st.sidebar.expander("Validation Metrics"):
        det_metrics = st.multiselect("Deterministic Validation Metrics", [
            "Root Mean Square Error",
            "Minkowski Distance"
        ])
        prob_metrics = st.multiselect("Probability-Based Validation Metrics", [
            "Normalized Euclidean Metric",
            "Mahalanobis Distance",
            "Kullback-Leibler Divergence",
            "Symmetrized Divergence",
            "Jensen-Shannon Divergence",
            "Hellinger Metric",
            "Kolmogorov-Smirnov Test",
            "Total Variation Distance"
        ])
        sig_metrics = st.multiselect("Signal Processing Validation Metrics", [
            "Simple Cross Correlation",
            "Normalized Cross Correlation",
            "Normalized Zero-Mean Sum of Squared Distances",
            "Moravec Correlation",
            "Index of Agreement",
            "Sprague-Geers Metric"
        ])

    all_metrics = det_metrics + prob_metrics + sig_metrics
    show_results = st.sidebar.button("Run Metrics")

    if show_results:
        st.header("📈 Selected Validation Metrics")
        results = {}

        if "Root Mean Square Error" in det_metrics:
            results["Root Mean Square Error"] = np.sqrt(np.mean((data1 - data2) ** 2))

        if "Minkowski Distance" in det_metrics:
            results["Minkowski Distance (p=3)"] = minkowski(data1, data2, 3)

        if "Normalized Euclidean Metric" in prob_metrics:
            results["Normalized Euclidean Metric"] = np.linalg.norm(data1 - data2) / len(data1)

        if "Mahalanobis Distance" in prob_metrics:
            cov = np.cov(np.stack([data1, data2]), rowvar=False)
            VI = inv(cov)
            diff = np.mean(data1) - np.mean(data2)
            results["Mahalanobis Distance"] = np.sqrt(diff ** 2 * VI[0, 0])

        if "Kullback-Leibler Divergence" in prob_metrics:
            p = np.histogram(data1, bins=30, density=True)[0] + 1e-10
            q = np.histogram(data2, bins=30, density=True)[0] + 1e-10
            results["Kullback-Leibler Divergence"] = entropy(p, q)

        if "Symmetrized Divergence" in prob_metrics:
            results["Symmetrized Divergence"] = entropy(p, q) + entropy(q, p)

        if "Jensen-Shannon Divergence" in prob_metrics:
            m = 0.5 * (p + q)
            results["Jensen-Shannon Divergence"] = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

        if "Hellinger Metric" in prob_metrics:
            results["Hellinger Metric"] = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

        if "Kolmogorov-Smirnov Test" in prob_metrics:
            stat, _ = ks_2samp(data1, data2)
            results["Kolmogorov-Smirnov Test"] = stat

        if "Total Variation Distance" in prob_metrics:
            results["Total Variation Distance"] = 0.5 * np.sum(np.abs(p - q))

        if "Simple Cross Correlation" in sig_metrics:
            results["Simple Cross Correlation"] = np.corrcoef(data1, data2)[0, 1]

        if "Normalized Cross Correlation" in sig_metrics:
            results["Normalized Cross Correlation"] = np.sum((data1 - np.mean(data1)) * (data2 - np.mean(data2))) / (len(data1) * np.std(data1) * np.std(data2))

        if "Normalized Zero-Mean Sum of Squared Distances" in sig_metrics:
            results["Normalized Zero-Mean Sum of Squared Distances"] = np.sum((data1 - np.mean(data1) - (data2 - np.mean(data2)))**2)

        if "Moravec Correlation" in sig_metrics:
            results["Moravec Correlation"] = np.sum(np.abs(np.roll(data1, 1) - data2))

        if "Index of Agreement" in sig_metrics:
            numerator = np.sum((data1 - data2) ** 2)
            denom = np.sum((np.abs(data1 - np.mean(data2)) + np.abs(data2 - np.mean(data2))) ** 2)
            results["Index of Agreement"] = 1 - numerator / denom

        if "Sprague-Geers Metric" in sig_metrics:
            A = np.sum(((data1 - data2) / data1) ** 2)
            B = np.sum(((data2 - data1) / data2) ** 2)
            results["Sprague-Geers Metric"] = 100 * np.sqrt((A + B) / (2 * len(data1)))

        for key, value in results.items():
            st.markdown(f"**{key}:** `{value:.4f}`")

    # === Plotting ===
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(data) + 1) / len(data)
        return x, y

    def save_plot(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        return buf.getvalue()

    st.subheader("📊 Histogram")
    fig1, ax1 = plt.subplots()
    ax1.hist(data1, bins=30, alpha=0.6, label="Sample 1", color="skyblue")
    ax1.hist(data2, bins=30, alpha=0.6, label="Sample 2", color="salmon")
    ax1.set_title("Histogram of Samples")
    ax1.legend()
    st.pyplot(fig1)
    st.download_button("🖼️ Download Histogram", save_plot(fig1), file_name="histogram.png", mime="image/png")

    st.subheader("📈 ECDF")
    x1, y1 = ecdf(data1)
    x2, y2 = ecdf(data2)
    fig2, ax2 = plt.subplots()
    ax2.step(x1, y1, where='post', label="Sample 1", color="blue")
    ax2.step(x2, y2, where='post', label="Sample 2", color="red")
    ax2.set_title("Empirical CDFs")
    ax2.legend()
    st.pyplot(fig2)
    st.download_button("🖼️ Download ECDF", save_plot(fig2), file_name="ecdf.png", mime="image/png")

    # === Summary Section ===
    with st.expander("ℹ️ Metric Descriptions"):
        st.markdown("""
        - **Root Mean Square Error (RMSE)**: Measures the average magnitude of the error.
        - **Minkowski Distance**: A general distance metric (p=3 used).
        - **Normalized Euclidean Metric**: Euclidean distance normalized by number of elements.
        - **Mahalanobis Distance**: Accounts for variance in each dimension.
        - **Kullback-Leibler Divergence**: Measures information loss when one distribution is used to approximate another.
        - **Symmetrized Divergence**: Sum of KL divergences in both directions.
        - **Jensen-Shannon Divergence**: Symmetric version of KL divergence.
        - **Hellinger Metric**: Measures similarity between two probability distributions.
        - **Kolmogorov-Smirnov Test**: Measures the maximum distance between ECDFs.
        - **Total Variation Distance**: Half of the L1 distance between two distributions.
        - **Simple Cross Correlation**: Measures linear correlation.
        - **Normalized Cross Correlation**: Cross correlation normalized by variance.
        - **Normalized Zero-Mean SSD**: Sum of squared differences after zero-meaning.
        - **Moravec Correlation**: Window-based correlation sensitive to texture.
        - **Index of Agreement**: Degree of model prediction accuracy.
        - **Sprague-Geers Metric**: Compares measured and predicted data magnitudes.
        """)

else:
    st.info("""
    📂 Please upload both CSV files to begin.

    If you don't have the CSV files yet, you can generate them using the **Generate Distribution** page on the left.
    
    To do so, navigate to the **Generate Distribution** page, select the desired distribution (Normal, Uniform, Exponential, or Binomial), 
    set the parameters, and then download the generated data as a CSV file. 
    Once you have your files, you can upload them here to compare distributions.
""")
