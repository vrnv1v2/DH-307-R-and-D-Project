import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="RSV Dashboard", layout="wide")
st.title("RSV Data Dashboard")

# ----------------------------
# Load Excel file
# ----------------------------
excel_file = "DH307_Dashboard_Code_V12.xlsx"
try:
    xls = pd.ExcelFile(excel_file)
except Exception as e:
    st.error(f"Failed to open Excel file: {e}")
    st.stop()

sheet_names = xls.sheet_names

# ----------------------------
# Sidebar: Variant & Feature
# ----------------------------
st.sidebar.header("Dashboard Controls")
variants = ["rsv_a", "rsv_b", "rsv_ab", "deaths"]  # added deaths
variant = st.sidebar.selectbox("Select Variant / Analysis", variants)

feature_name_map = {
    "Region": "Geographic Analysis",
    "State": "State-wise Analysis",
    "AgeBand": "Age-wise Analysis",
    "Socio-Demographic Status": "Socio-Demographic Status",
    "sregion": "Symptoms vs Region",
    "symptom": "Symptoms vs Variant",
    "serious": "Serious Diseases vs RSV",
    "season": "Seasonal Analysis",
    "vaccine": "Vaccine Analysis"
}

# Add variant sheets dynamically
if variant != "deaths":
    variant_sheets = [s for s in sheet_names if s.startswith(f"{variant}_") and not s.endswith("_nregion")]
    features = [s.replace(f"{variant}_", "") for s in variant_sheets]
    friendly_features = [feature_name_map.get(f, f) for f in features]

    # Include vaccine if exists
    vaccine_sheet_name = f"{variant}_vaccine"
    if vaccine_sheet_name in sheet_names and "Vaccine Analysis" not in friendly_features:
        friendly_features.append("Vaccine Analysis")
        features.append("vaccine")

    feature_friendly = st.sidebar.selectbox("Select Feature / Analysis", friendly_features)
    feature_index = friendly_features.index(feature_friendly)
    sheet_to_load = f"{variant}_{features[feature_index]}"
else:
    feature_friendly = None
    sheet_to_load = "deaths"

# ----------------------------
# Load selected sheet
# ----------------------------
try:
    df = pd.read_excel(excel_file, sheet_name=sheet_to_load)
except Exception as e:
    st.error(f"Failed to load sheet '{sheet_to_load}': {e}")
    st.stop()
df.columns = df.columns.str.strip()

# ----------------------------
# Deaths-specific preprocessing
# ----------------------------
if variant == "deaths":
    df["Yes"] = pd.to_numeric(df["Yes"].astype(str).str.replace(",", "").str.strip(), errors="coerce").fillna(0)
    df["No"] = pd.to_numeric(df["No"].astype(str).str.replace(",", "").str.strip(), errors="coerce").fillna(0)
    df = df[(df["Yes"] + df["No"]) > 0].copy()
    df["Total"] = df["Yes"] + df["No"]

# ----------------------------
# Compute % Positive if Yes/No exist
# ----------------------------
if variant != "deaths" and "Yes" in df.columns and "No" in df.columns:
    df["Yes"] = pd.to_numeric(df["Yes"].astype(str).str.replace(",", "").str.strip(), errors="coerce").fillna(0)
    df["No"] = pd.to_numeric(df["No"].astype(str).str.replace(",", "").str.strip(), errors="coerce").fillna(0)
    df = df[(df["Yes"] + df["No"]) > 0].copy()
    df["% Positive"] = (df["Yes"] / (df["Yes"] + df["No"])) * 100
    df["% Positive"] = df["% Positive"].round(2)

geojson_path = "india_state.geojson"
try:
    gdf = gpd.read_file(geojson_path)
    gdf['NAME_1_lower'] = gdf['NAME_1'].str.strip().str.lower()
except Exception as e:
    st.error(f"Failed to load local India GeoJSON: {e}")
    st.stop()

# ----------------------------
# Normalize State column
# ----------------------------
if variant !='death' and 'State' in df.columns:
    df['State_normalized'] = df['State'].astype(str).str.strip().str.lower()
    state_corrections = {
        "jammu kashmir": "jammu & kashmir",
        "uttar pradesh": "uttar pradesh",
        "madhya pradesh": "madhya pradesh",
        "tamil nadu": "tamil nadu",
        "west bengal": "west bengal",
        "kerala": "kerala",
        "gujarat": "gujarat",
        "maharashtra": "maharashtra",
        "chhattisgarh": "chhattisgarh"
    }
    df['State_normalized'] = df['State_normalized'].replace(state_corrections)

# ----------------------------
# Plotting functions
# ----------------------------
def plot_state_map(df):
    if "% Positive" not in df.columns or "State_normalized" not in df.columns:
        st.info("No State Yes/No data available for map.")
        return

    df_plot = df[df['State_normalized'].isin(gdf['NAME_1_lower'])].copy()
    if df_plot.empty:
        st.info("No valid states found in map.")
        return

    gdf_merged = gdf.merge(
        df_plot[['State_normalized', '% Positive']],
        left_on='NAME_1_lower',
        right_on='State_normalized',
        how='left'
    )

    fig, ax = plt.subplots(figsize=(12, 12))
    gdf_merged.plot(
        column="% Positive",
        cmap="YlOrRd",
        linewidth=0.8,
        ax=ax,
        edgecolor="0.8",
        legend=True,
        vmin=0,
        vmax=df_plot["% Positive"].max() * 1.1
    )
    ax.set_title(f"{variant.upper()} - State-wise RSV % Positive", fontsize=16)
    ax.axis('off')
    st.pyplot(fig)

    top_state = df_plot.sort_values("% Positive", ascending=False).iloc[0]
    st.markdown(f"*Top State:* {top_state['State']} ({top_state['% Positive']:.2f}%)")

def plot_pie_chart(df, feature_col):
    if "% Positive" not in df.columns:
        st.info("No Yes/No data available for this feature.")
        return
    df_plot = df.copy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(df_plot["% Positive"], labels=df_plot[feature_col], autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors)
    ax.set_title(f"% Positive ({variant.upper()}) - {feature_friendly}", fontsize=14, fontweight='bold')
    st.pyplot(fig)

# ----------------------------
# Combined RSV Positive & Negative sregion plots
# ----------------------------
def plot_sregion_combined(variant, excel_file):
    s_sheet = f"{variant}_sregion"
    n_sheet = f"{variant}_nregion"

    try:
        s_df = pd.read_excel(excel_file, sheet_name=s_sheet)
        n_df = pd.read_excel(excel_file, sheet_name=n_sheet)
    except Exception as e:
        st.error(f"Failed to load sheets: {e}")
        return

    def preprocess(df):
        df = df.copy()
        df["Yes"] = pd.to_numeric(df["Yes"], errors="coerce").fillna(0)
        df["No"] = pd.to_numeric(df["No"], errors="coerce").fillna(0)
        df = df[(df["Yes"] + df["No"]) > 0]
        df["% Positive"] = (df["Yes"] / (df["Yes"] + df["No"])) * 100
        df["% Positive"] = df["% Positive"].round(2)
        df["Region"] = df["Region"].astype(str).str.strip()
        df["Symptom"] = df["Symptom"].astype(str).str.strip()
        return df

    s_df = preprocess(s_df)
    n_df = preprocess(n_df)
    all_symptoms = sorted(set(s_df["Symptom"]) | set(n_df["Symptom"]))
    cmap = plt.cm.tab20
    color_map = {symptom: cmap(i % cmap.N) for i, symptom in enumerate(all_symptoms)}
    regions = sorted(set(s_df["Region"]) | set(n_df["Region"]))

    for region in regions:
        s_subset = s_df[s_df["Region"] == region].sort_values("% Positive", ascending=False)
        n_subset = n_df[n_df["Region"] == region].sort_values("% Positive", ascending=False)

        if s_subset.empty and n_subset.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

        y_max = max(
            s_subset["% Positive"].max() if not s_subset.empty else 0,
            n_subset["% Positive"].max() if not n_subset.empty else 0,
            10
        ) * 1.1

        # RSV Positive
        if not s_subset.empty:
            colors = [color_map[sym] for sym in s_subset["Symptom"]]
            axes[0].bar(s_subset["Symptom"], s_subset["% Positive"], color=colors)
            axes[0].set_xticklabels(s_subset["Symptom"], rotation=45, ha="right")
            axes[0].set_ylim(0, y_max)
            axes[0].set_title(f"{region} - RSV Positive")
            axes[0].set_ylabel("% Positive")
            axes[0].grid(axis="y", linestyle="--", alpha=0.3)
        else:
            axes[0].set_visible(False)

        # RSV Negative
        if not n_subset.empty:
            colors = [color_map[sym] for sym in n_subset["Symptom"]]
            axes[1].bar(n_subset["Symptom"], n_subset["% Positive"], color=colors)
            axes[1].set_xticklabels(n_subset["Symptom"], rotation=45, ha="right")
            axes[1].set_ylim(0, y_max)
            axes[1].set_title(f"{region} - RSV Negative")
            axes[1].grid(axis="y", linestyle="--", alpha=0.3)
        else:
            axes[1].set_visible(False)

        plt.suptitle(f"Symptoms by Region - {region}", fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)

        # Top symptoms
        if not s_subset.empty:
            top_symptom = s_subset.iloc[0]
            st.markdown(f"*RSV Positive - Top Symptom:* {top_symptom['Symptom']} ({top_symptom['% Positive']:.2f}%)")
        if not n_subset.empty:
            top_symptom_n = n_subset.iloc[0]
            st.markdown(f"*RSV Negative - Top Symptom:* {top_symptom_n['Symptom']} ({top_symptom_n['% Positive']:.2f}%)")
# ----------------------------
# Combined RSV Positive & Negative Symptoms vs Variant plots
# ----------------------------
def plot_symptoms_variant_combined(variant, excel_file):
    pos_sheet = f"{variant}_RSV Positive"
    neg_sheet = f"{variant}_RSV Negative"

    try:
        pos_df = pd.read_excel(excel_file, sheet_name=pos_sheet)
        neg_df = pd.read_excel(excel_file, sheet_name=neg_sheet)
    except Exception as e:
        st.error(f"Failed to load sheets: {e}")
        return

    def preprocess(df):
        df = df.copy()
        df["Yes"] = pd.to_numeric(df["Yes"], errors="coerce").fillna(0)
        df["No"] = pd.to_numeric(df["No"], errors="coerce").fillna(0)
        df = df[(df["Yes"] + df["No"]) > 0]
        df["% Positive"] = (df["Yes"] / (df["Yes"] + df["No"])) * 100
        df["% Positive"] = df["% Positive"].round(2)
        df["Symptom"] = df["Symptom"].astype(str).str.strip()
        return df

    pos_df = preprocess(pos_df)
    neg_df = preprocess(neg_df)

    symptoms = sorted(set(pos_df["Symptom"].unique()) | set(neg_df["Symptom"].unique()))

    for symptom in symptoms:
        pos_subset = pos_df[pos_df["Symptom"] == symptom]
        neg_subset = neg_df[neg_df["Symptom"] == symptom]

        if pos_subset.empty and neg_subset.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        y_max = max(
            pos_subset["% Positive"].max() if not pos_subset.empty else 0,
            neg_subset["% Positive"].max() if not neg_subset.empty else 0,
            10
        ) * 1.1

        # RSV Positive
        if not pos_subset.empty:
            axes[0].bar(pos_subset["Symptom"], pos_subset["% Positive"],
                        color=plt.cm.viridis(pos_subset["% Positive"]/100))
            axes[0].set_xticklabels(pos_subset["Symptom"], rotation=45, ha="right")
            axes[0].set_ylim(0, y_max)
            axes[0].set_title(f"Symptoms vs Variant - RSV Positive", fontsize=14)
            axes[0].set_ylabel("% Positive")
            axes[0].grid(axis="y", linestyle="--", alpha=0.3)
        else:
            axes[0].set_visible(False)

        # RSV Negative
        if not neg_subset.empty:
            axes[1].bar(neg_subset["Symptom"], neg_subset["% Positive"],
                        color=plt.cm.viridis(neg_subset["% Positive"]/100))
            axes[1].set_xticklabels(neg_subset["Symptom"], rotation=45, ha="right")
            axes[1].set_ylim(0, y_max)
            axes[0].set_title(f"Symptoms vs Variant - RSV Negative", fontsize=14)
            axes[1].grid(axis="y", linestyle="--", alpha=0.3)
        else:
            axes[1].set_visible(False)

        plt.suptitle(f"Symptoms vs Variant - {symptom}", fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)

        # Top symptom info
        if not pos_subset.empty:
            st.markdown(f"*RSV Positive - {symptom}:* {pos_subset.iloc[0]['Symptom']} ({pos_subset.iloc[0]['% Positive']:.2f}%)")
        if not neg_subset.empty:
            st.markdown(f"*RSV Negative - {symptom}:* {neg_subset.iloc[0]['Symptom']} ({neg_subset.iloc[0]['% Positive']:.2f}%)")

# ----------------------------
# Other generic bar chart
# ----------------------------
def plot_bar_chart(df, feature_col):
    if "% Positive" not in df.columns:
        st.info("No Yes/No data available for this feature.")
        return

    df_plot = df.copy()
    df_plot[feature_col] = df_plot[feature_col].astype(str).str.strip()
    df_plot = df_plot[df_plot[feature_col] != ""].copy()
    df_plot["% Positive"] = pd.to_numeric(df_plot["% Positive"], errors="coerce").fillna(0)
    df_plot = df_plot.sort_values("% Positive", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(df_plot)), df_plot["% Positive"],
                   color=plt.cm.viridis(df_plot["% Positive"] / df_plot["% Positive"].max()))
    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot[feature_col], rotation=45, ha='right')
    ax.set_xlabel(feature_col, fontsize=12)
    ax.set_ylabel("% Positive", fontsize=12)
    ax.set_title(f"{variant.upper()} - {feature_friendly}", fontsize=14, fontweight='bold')
    ax.set_ylim(0, df_plot["% Positive"].max() * 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
def plot_symptom_difference(variant, excel_file):
    pos_sheet = f"{variant}_RSV Positive"
    neg_sheet = f"{variant}_RSV Negative"

    try:
        pos_df = pd.read_excel(excel_file, sheet_name=pos_sheet)
        neg_df = pd.read_excel(excel_file, sheet_name=neg_sheet)
    except Exception as e:
        st.error(f"Failed to load sheets: {e}")
        return

    # Clean and convert numeric columns
    for df in [pos_df, neg_df]:
        df["Symptom"] = df["Symptom"].astype(str).str.strip()
        df["Yes"] = pd.to_numeric(df["Yes"], errors="coerce").fillna(0)

    # Merge on Symptom
    merged = pd.merge(pos_df[["Symptom", "Yes"]], neg_df[["Symptom", "Yes"]],
                      on="Symptom", how="outer", suffixes=("_pos", "_neg")).fillna(0)

    merged["Difference"] = merged["Yes_pos"] - merged["Yes_neg"]
    merged = merged.sort_values("Difference", ascending=False).reset_index(drop=True)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1b9e77" if x > 0 else "#d95f02" for x in merged["Difference"]]
    ax.bar(merged["Symptom"], merged["Difference"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticklabels(merged["Symptom"], rotation=45, ha="right")
    ax.set_ylabel("Difference in Yes Counts (Positive − Negative)")
    ax.set_title(f"{variant.upper()} - Symptom Difference (RSV Positive − Negative)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Highlight top and bottom differences
    top = merged.iloc[0]
    bottom = merged.iloc[-1]
    st.markdown(f"**Most overrepresented in RSV Positive:** {top['Symptom']} (+{int(top['Difference'])})")
    st.markdown(f"**Most overrepresented in RSV Negative:** {bottom['Symptom']} ({int(bottom['Difference'])})")

# ----------------------------
# Seasonal & Vaccine
# ----------------------------
def plot_seasonal(df):
    if 'Month of Onset' not in df.columns or 'Cases' not in df.columns:
        st.info("Seasonal sheet missing required columns.")
        return

    df = df.copy()
    df['Cases'] = pd.to_numeric(df['Cases'], errors='coerce').fillna(0)
    df['Month'] = pd.to_datetime(df['Month of Onset'], format='%b-%y', errors='coerce')
    df = df.dropna(subset=['Month'])
    df['Year'] = df['Month'].dt.year
    df['Month_Num'] = df['Month'].dt.month

    # Prepare pivot for comparison
    pivot_df = df.pivot_table(values='Cases', index='Month_Num', columns='Year', aggfunc='sum').fillna(0)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig, ax = plt.subplots(figsize=(12, 6))
    for year in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[year], marker='o', label=str(year))

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, rotation=45, ha='right')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('RSV Cases', fontsize=12)
    ax.set_title(f"{variant.upper()} - Seasonal Comparison (2023 vs 2024)", fontsize=14, fontweight='bold')
    ax.legend(title="Year")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


def plot_vaccine_analysis(df):
    if not all(col in df.columns for col in ["Vaccine", "Status", "Yes", "No"]):
        st.info("Vaccine sheet missing required columns.")
        return

    df["Yes"] = pd.to_numeric(df["Yes"], errors="coerce").fillna(0)
    df["No"] = pd.to_numeric(df["No"], errors="coerce").fillna(0)
    df["% Positive"] = (df["Yes"] / (df["Yes"] + df["No"])) * 100

    vaccines = df["Vaccine"].unique()
    x = range(len(vaccines))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    yes_values = df[df["Status"]=="Yes"].set_index("Vaccine")["% Positive"].reindex(vaccines)
    no_values = df[df["Status"]=="No"].set_index("Vaccine")["% Positive"].reindex(vaccines)

    ax.bar([i - width/2 for i in x], no_values, width=width, label="No")
    ax.bar([i + width/2 for i in x], yes_values, width=width, label="Yes")

    ax.set_xticks(x)
    ax.set_xticklabels(vaccines, rotation=45, ha='right')
    ax.set_ylabel("% RSV Positive")
    ax.set_title("RSV % Positive by Vaccine Status")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
def plot_deaths_bars(df):
    df_plot = df.copy()
    features = df_plot["Symptom"].tolist()
    yes_vals = df_plot["Yes"].tolist()
    no_vals = df_plot["No"].tolist()
    batch_size = 11  # 10-15 features per plot

    for i in range(0, len(features), batch_size):
        f_batch = features[i:i+batch_size]
        yes_batch = yes_vals[i:i+batch_size]
        no_batch = no_vals[i:i+batch_size]

        x = np.arange(len(f_batch))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, yes_batch, width, label="Deaths (Yes)", color="red")
        ax.bar(x + width/2, no_batch, width, label="Deaths (No)", color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(f_batch, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Deaths by Symptom")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
# ----------------------------
# Tabs
# ----------------------------
tabs = st.tabs(["Plot", "Data Table"])

with tabs[0]:
    if variant == "deaths":
        plot_deaths_bars(df)
    else:
        feature = features[feature_index].lower()
        if feature == "state":
            plot_state_map(df)
        elif feature == "region":
            plot_pie_chart(df, "Region")
        elif feature == "sregion":
            plot_sregion_combined(variant, excel_file)
        elif "symptom" in feature:
            with st.container():
                plot_symptoms_variant_combined(variant, excel_file)
                st.markdown("---")
                st.subheader("Difference in Symptom Prevalence (RSV Positive − Negative)")
                plot_symptom_difference(variant, excel_file)


        elif feature == "season":
            plot_seasonal(df)
        elif feature == "vaccine":
            plot_vaccine_analysis(df)
        else:
            feature_cols = [c for c in df.columns if c not in ["Yes", "No", "% Positive", "State_normalized"]]
            if feature_cols:
                plot_bar_chart(df, feature_cols[0])


with tabs[1]:
    st.subheader("Full Data Table")
    st.dataframe(df, use_container_width=True)
