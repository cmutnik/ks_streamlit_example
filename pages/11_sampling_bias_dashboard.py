# Copyright (c) 2025 takotime808

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sampling Bias Detection", layout="wide")
st.title("\U0001F4CA Sampling Bias Detection Dashboard")

uploaded_file = st.file_uploader("Upload your regression dataset (CSV)", type="csv")


def plot_pairgrid_with_kde(df, num_cols):
    st.subheader("Pairwise Relationships (KDE on Lower Triangle)")
    if len(num_cols) < 2:
        st.warning("Need at least two numeric columns for pairwise visualization.")
        return None
    g = sns.PairGrid(df[num_cols])
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot, cmap="Blues", fill=True)
    g.map_diag(sns.histplot, edgecolor="black")
    plt.tight_layout()
    return g


def plot_corr_heatmap(df, num_cols):
    st.subheader("Correlation Heatmap")
    corr = df[num_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


def plot_histograms(df, num_cols):
    st.subheader("Feature Distributions")
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(col)
        st.pyplot(fig)


def infer_sampling(df, num_cols):
    n_unique_vals = [len(df[col].unique()) for col in num_cols]
    grid_like = np.prod(n_unique_vals) == len(df)
    lhs_like = all(df[col].value_counts().max() == 1 for col in num_cols)
    if grid_like:
        st.info(
            "Sampling looks like grid sampling (regular mesh in value pairs, distinct fixed values per feature)."
        )
    elif lhs_like:
        st.info(
            "Sampling looks like Latin hypercube sampling (one value per bin per feature, uniform marginals)."
        )
    else:
        st.info(
            "Sampling does not match grid or LHS. It may be a random or other custom scheme."
        )


def main():
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            st.warning("Need at least two numeric columns for sampling pattern analysis.")
            return
        plot_histograms(df, num_cols)
        pairgrid = plot_pairgrid_with_kde(df, num_cols)
        if pairgrid:
            st.pyplot(pairgrid.fig)
        plot_corr_heatmap(df, num_cols)
        infer_sampling(df, num_cols)
    else:
        st.info("Please upload a CSV file with your regression data.")


if __name__ == "__main__":
    main()