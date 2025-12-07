# app.py
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="School Performance & Entrant Visualisations",
    layout="wide",
)

# =========================
# FILE LOCATIONS
# =========================
BASE_DIR = Path(__file__).parent

FILE_OCC = BASE_DIR / "Count by Occupation.xlsx"
FILE_LEVEL = BASE_DIR / "Count by Level of Qualification.xlsx"
FILE_HECOS = BASE_DIR / "HECoS Data.xlsx"
FILE_VIZ = BASE_DIR / "visualizations.csv"


# =========================
# HELPERS
# =========================
@st.cache_data
def load_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def multiselect_filter(df, col, selected):
    if col not in df.columns:
        return df
    if not selected or "All" in selected:
        return df
    return df[df[col].isin(selected)]


# =========================
# LOAD DATA (WITH BASIC ERRORS)
# =========================
dfs = {}

with st.sidebar:
    st.header("Data sources")

    for name, path in [
        ("Count by Occupation", FILE_OCC),
        ("Count by Level of Qualification", FILE_LEVEL),
        ("HECoS Data", FILE_HECOS),
        ("Visualizations (regression data)", FILE_VIZ),
    ]:
        if path.exists():
            st.success(f"Found: {path.name}")
        else:
            st.error(f"Missing: {path.name}")

try:
    dfs["occ"] = load_excel(FILE_OCC)
except Exception as e:
    dfs["occ_error"] = str(e)

try:
    dfs["level"] = load_excel(FILE_LEVEL)
except Exception as e:
    dfs["level_error"] = str(e)

try:
    dfs["hecos"] = load_excel(FILE_HECOS)
except Exception as e:
    dfs["hecos_error"] = str(e)

try:
    dfs["viz"] = load_csv(FILE_VIZ)
except Exception as e:
    dfs["viz_error"] = str(e)


# =========================
# LAYOUT
# =========================
st.title("📊 School Performance & Entrant Visualisations")

st.markdown(
    """
Use the tabs below to explore:

1. **Count by Occupation** – socio-economic / category breakdowns  
2. **Count by Level of Qualification** – time trends by qualification level  
3. **HECoS Subject Data** – subject-level entrants by CAH grouping  
4. **Regression Visualisations** – A8, EBacc, Progress 8, thresholds (from `visualizations.csv`)
"""
)

tab_occ, tab_level, tab_hecos, tab_viz = st.tabs(
    [
        "Count by Occupation",
        "Count by Level of Qualification",
        "HECoS Subject Data",
        "Regression Visualizations",
    ]
)

# =========================
# TAB 1 – COUNT BY OCCUPATION
# =========================
with tab_occ:
    st.header("Count by Occupation")

    if "occ_error" in dfs:
        st.error(f"Could not load Count by Occupation.xlsx: {dfs['occ_error']}")
    else:
        df = dfs["occ"].copy()
        st.caption("Source: Count by Occupation.xlsx")

        # Basic column sanity
        # Expected columns (adjust if your file differs)
        # 'Category Marker', 'Country of HE provider', 'Entrant marker',
        # 'Level of study', 'Academic Year', 'Category', 'Number', 'Percentage'
        with st.expander("Data preview", expanded=False):
            st.dataframe(df.head(), use_container_width=True)

        st.markdown("### Filters")

        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)

        cat_markers = sorted(df["Category Marker"].dropna().unique()) if "Category Marker" in df.columns else []
        cat_marker_sel = col1.multiselect(
            "Category marker",
            options=cat_markers,
            default=cat_markers,
            key="occ_cat_marker",
        )

        he_countries = sorted(df["Country of HE provider"].dropna().unique()) if "Country of HE provider" in df.columns else []
        he_country_sel = col2.multiselect(
            "Country of HE provider",
            options=he_countries,
            default=["England"] if "England" in he_countries else he_countries,
            key="occ_country",
        )

        entrant_vals = sorted(df["Entrant marker"].dropna().unique()) if "Entrant marker" in df.columns else []
        entrant_sel = col3.multiselect(
            "Entrant marker",
            options=entrant_vals,
            default=entrant_vals,
            key="occ_entrant",
        )

        level_vals = sorted(df["Level of study"].dropna().unique()) if "Level of study" in df.columns else []
        level_sel = col4.multiselect(
            "Level of study",
            options=level_vals,
            default=level_vals,
            key="occ_level",
        )

        year_vals = sorted(df["Academic Year"].dropna().unique()) if "Academic Year" in df.columns else []
        year_sel = col5.multiselect(
            "Academic Year",
            options=year_vals,
            default=year_vals,
            key="occ_year",
        )

        df_f = df.copy()
        df_f = multiselect_filter(df_f, "Category Marker", cat_marker_sel)
        df_f = multiselect_filter(df_f, "Country of HE provider", he_country_sel)
        df_f = multiselect_filter(df_f, "Entrant marker", entrant_sel)
        df_f = multiselect_filter(df_f, "Level of study", level_sel)
        if "Academic Year" in df_f.columns and year_sel:
            df_f = df_f[df_f["Academic Year"].isin(year_sel)]

        metric = st.radio(
            "Metric",
            ["Number", "Percentage"],
            horizontal=True,
            key="occ_metric",
        )

        st.markdown("### Chart")

        if df_f.empty:
            st.warning("No data after filters. Adjust your selections.")
        else:
            if metric not in df_f.columns:
                st.error(f"Column '{metric}' not found in data.")
            else:
                # Group by category and year
                if "Category" not in df_f.columns or "Academic Year" not in df_f.columns:
                    st.error("Expected 'Category' and 'Academic Year' columns not found.")
                else:
                    agg = (
                        df_f.groupby(["Academic Year", "Category"], as_index=False)[metric]
                        .sum()
                    )

                    view = st.radio(
                        "View",
                        ["By Category (latest year)", "Trend over time (by category)"],
                        horizontal=True,
                        key="occ_view",
                    )

                    if view == "By Category (latest year)":
                        latest_year = sorted(agg["Academic Year"].unique())[-1]
                        latest = agg[agg["Academic Year"] == latest_year]
                        latest = latest.sort_values(metric, ascending=False)

                        st.caption(f"Latest year in filtered data: **{latest_year}**")
                        chart_df = latest.set_index("Category")[metric]
                        st.bar_chart(chart_df, use_container_width=True)
                    else:
                        pivot = agg.pivot(
                            index="Academic Year", columns="Category", values=metric
                        ).sort_index()
                        st.line_chart(pivot, use_container_width=True)

        with st.expander("Filtered data"):
            st.dataframe(df_f, use_container_width=True, height=300)


# =========================
# TAB 2 – COUNT BY LEVEL OF QUALIFICATION
# =========================
with tab_level:
    st.header("Count by Level of Qualification")

    if "level_error" in dfs:
        st.error(f"Could not load Count by Level of Qualification.xlsx: {dfs['level_error']}")
    else:
        df = dfs["level"].copy()
        st.caption("Source: Count by Level of Qualification.xlsx")

        with st.expander("Data preview", expanded=False):
            st.dataframe(df.head(), use_container_width=True)

        # Expected columns:
        # 'Level of qualification', 'Academic year', 'Number'
        level_vals = sorted(df["Level of qualification"].dropna().unique()) if "Level of qualification" in df.columns else []
        year_vals = sorted(df["Academic year"].dropna().unique()) if "Academic year" in df.columns else []

        col1, col2 = st.columns(2)

        level_sel = col1.multiselect(
            "Level of qualification",
            options=level_vals,
            default=[lv for lv in level_vals if lv.lower() != "total"] or level_vals,
            key="level_level_sel",
        )

        year_sel = col2.multiselect(
            "Academic year",
            options=year_vals,
            default=year_vals,
            key="level_year_sel",
        )

        df_f = df.copy()
        df_f = multiselect_filter(df_f, "Level of qualification", level_sel)
        if "Academic year" in df_f.columns and year_sel:
            df_f = df_f[df_f["Academic year"].isin(year_sel)]

        if df_f.empty:
            st.warning("No data after filters.")
        else:
            if "Number" not in df_f.columns:
                st.error("Expected 'Number' column not found.")
            else:
                st.markdown("### Time series of entrants by level of qualification")

                pivot = (
                    df_f.pivot(
                        index="Academic year",
                        columns="Level of qualification",
                        values="Number",
                    )
                    .sort_index()
                )
                st.line_chart(pivot, use_container_width=True)

                st.markdown("### Composition by level (latest year)")
                latest_year = sorted(df_f["Academic year"].unique())[-1]
                latest = df_f[df_f["Academic year"] == latest_year]
                latest = latest.groupby("Level of qualification", as_index=False)["Number"].sum()
                latest = latest.sort_values("Number", ascending=False)
                st.caption(f"Latest year in filtered data: **{latest_year}**")

                chart_df = latest.set_index("Level of qualification")["Number"]
                st.bar_chart(chart_df, use_container_width=True)

        with st.expander("Filtered data"):
            st.dataframe(df_f, use_container_width=True, height=300)


# =========================
# TAB 3 – HECOS / SUBJECT DATA
# =========================
with tab_hecos:
    st.header("HECoS / CAH Subject Data")

    if "hecos_error" in dfs:
        st.error(f"Could not load HECoS Data.xlsx: {dfs['hecos_error']}")
    else:
        df = dfs["hecos"].copy()
        st.caption("Source: HECoS Data.xlsx")

        with st.expander("Data preview", expanded=False):
            st.dataframe(df.head(), use_container_width=True)

        # Expected columns (adjust if needed):
        # 'CAH level marker', 'Entrant marker', 'Level of study',
        # 'Mode of study', 'Academic Year', 'CAH level subject', 'Number'
        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)

        cah_markers = sorted(df["CAH level marker"].dropna().unique()) if "CAH level marker" in df.columns else []
        cah_sel = col1.multiselect(
            "CAH level marker",
            options=cah_markers,
            default=cah_markers,
            key="hecos_cah_sel",
        )

        entrant_vals = sorted(df["Entrant marker"].dropna().unique()) if "Entrant marker" in df.columns else []
        entrant_sel = col2.multiselect(
            "Entrant marker",
            options=entrant_vals,
            default=entrant_vals,
            key="hecos_entrant_sel",
        )

        level_vals = sorted(df["Level of study"].dropna().unique()) if "Level of study" in df.columns else []
        level_sel = col3.multiselect(
            "Level of study",
            options=level_vals,
            default=level_vals,
            key="hecos_level_sel",
        )

        mode_vals = sorted(df["Mode of study"].dropna().unique()) if "Mode of study" in df.columns else []
        mode_sel = col4.multiselect(
            "Mode of study",
            options=mode_vals,
            default=mode_vals,
            key="hecos_mode_sel",
        )

        year_vals = sorted(df["Academic Year"].dropna().unique()) if "Academic Year" in df.columns else []
        year_sel = col5.multiselect(
            "Academic Year",
            options=year_vals,
            default=year_vals,
            key="hecos_year_sel",
        )

        df_f = df.copy()
        df_f = multiselect_filter(df_f, "CAH level marker", cah_sel)
        df_f = multiselect_filter(df_f, "Entrant marker", entrant_sel)
        df_f = multiselect_filter(df_f, "Level of study", level_sel)
        df_f = multiselect_filter(df_f, "Mode of study", mode_sel)
        if "Academic Year" in df_f.columns and year_sel:
            df_f = df_f[df_f["Academic Year"].isin(year_sel)]

        if df_f.empty:
            st.warning("No data after filters.")
        else:
            if "Number" not in df_f.columns or "CAH level subject" not in df_f.columns:
                st.error("Expected 'Number' and 'CAH level subject' columns not found.")
            else:
                st.markdown("### Entrants by subject (latest year)")

                latest_year = sorted(df_f["Academic Year"].unique())[-1]
                latest = df_f[df_f["Academic Year"] == latest_year]
                latest = (
                    latest.groupby("CAH level subject", as_index=False)["Number"]
                    .sum()
                    .sort_values("Number", ascending=False)
                )

                st.caption(f"Latest year in filtered data: **{latest_year}**")
                top_n = st.slider(
                    "Top N subjects",
                    min_value=5,
                    max_value=40,
                    value=20,
                    key="hecos_top_n",
                )

                latest_top = latest.head(top_n)
                chart_df = latest_top.set_index("CAH level subject")["Number"]
                st.bar_chart(chart_df, use_container_width=True)

                st.markdown("### Trend for a selected subject over time")
                subject_sel = st.selectbox(
                    "Subject",
                    options=sorted(df_f["CAH level subject"].unique()),
                    key="hecos_subject",
                )

                subj_ts = (
                    df_f[df_f["CAH level subject"] == subject_sel]
                    .groupby("Academic Year", as_index=False)["Number"]
                    .sum()
                    .sort_values("Academic Year")
                )

                subj_ts = subj_ts.set_index("Academic Year")
                st.line_chart(subj_ts["Number"], use_container_width=True)

        with st.expander("Filtered data"):
            st.dataframe(df_f, use_container_width=True, height=300)


# =========================
# TAB 4 – REGRESSION VISUALIZATIONS (visualizations.csv)
# =========================
with tab_viz:
    st.header("Regression Visualisations (from visualizations.csv)")

    if "viz_error" in dfs:
        st.error(f"Could not load visualizations.csv: {dfs['viz_error']}")
    else:
        df = dfs["viz"].copy()
        st.caption("Source: visualizations.csv")

        with st.expander("Data preview", expanded=False):
            st.dataframe(df.head(), use_container_width=True)

        # Expected columns from your regression script:
        # school_urn, school_name, la_name, time_period,
        # avg_att8, avg_ebaccaps, avg_p8score, pt_anypass, pt_entbasics
        num_cols = [c for c in df.columns if df[c].dtype != "object"]
        all_cols = df.columns.tolist()

        # Sidebar-style filters inside tab
        st.markdown("### Filters")

        col1, col2 = st.columns(2)

        la_vals = sorted(df["la_name"].dropna().unique()) if "la_name" in df.columns else []
        la_sel = col1.multiselect(
            "Local authority (la_name)",
            options=la_vals,
            default=la_vals,
            key="viz_la_sel",
        )

        tp_vals = sorted(df["time_period"].dropna().unique()) if "time_period" in df.columns else []
        tp_sel = col2.multiselect(
            "Time period",
            options=tp_vals,
            default=tp_vals,
            key="viz_tp_sel",
        )

        df_f = df.copy()
        df_f = multiselect_filter(df_f, "la_name", la_sel)
        df_f = multiselect_filter(df_f, "time_period", tp_sel)

        if df_f.empty:
            st.warning("No data after filters.")
        else:
            st.markdown("### Scatter: Progress vs Attainment 8")

            if all(col in df_f.columns for col in ["avg_p8score", "avg_att8"]):
                scatter_df = df_f[["avg_p8score", "avg_att8"]].copy()
                st.scatter_chart(
                    scatter_df.rename(
                        columns={
                            "avg_p8score": "Progress 8",
                            "avg_att8": "Attainment 8",
                        }
                    ),
                    x="Progress 8",
                    y="Attainment 8",
                )
            else:
                st.info("Columns 'avg_p8score' and/or 'avg_att8' not found.")

            st.markdown("### Scatter: Progress vs EBacc APS")

            if all(col in df_f.columns for col in ["avg_p8score", "avg_ebaccaps"]):
                scatter_df2 = df_f[["avg_p8score", "avg_ebaccaps"]].copy()
                st.scatter_chart(
                    scatter_df2.rename(
                        columns={
                            "avg_p8score": "Progress 8",
                            "avg_ebaccaps": "EBacc APS",
                        }
                    ),
                    x="Progress 8",
                    y="EBacc APS",
                )
            else:
                st.info("Columns 'avg_p8score' and/or 'avg_ebaccaps' not found.")

            st.markdown("### LA-level average outcomes")

            if all(col in df_f.columns for col in ["la_name", "avg_att8", "avg_ebaccaps"]):
                la_group = (
                    df_f.groupby("la_name", as_index=False)[["avg_att8", "avg_ebaccaps", "avg_p8score"]]
                    .mean()
                )
                la_group = la_group.sort_values("avg_att8", ascending=False)

                metric_choice = st.selectbox(
                    "Metric to plot by LA",
                    options=["avg_att8", "avg_ebaccaps", "avg_p8score"],
                    index=0,
                    key="viz_la_metric",
                )

                chart_df = la_group.set_index("la_name")[metric_choice]
                st.bar_chart(chart_df, use_container_width=True)
            else:
                st.info("LA or outcomes columns not found for LA-level summary.")

        with st.expander("Filtered data"):
            st.dataframe(df_f, use_container_width=True, height=300)
