import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- Configuration ---
PIE_GROUPS = {"Andesitic Eruption Style (Next Eruption)": [5, 6, 7, 8, 9]}


@st.cache_data
def load_all_data(elicitation_path):
    """
    Load all available CSV files for a given elicitation.

    Parameters
    ----------
    elicitation_path : pathlib.Path
        Path to the elicitation directory.

    Returns
    -------
    dict or None
        Dictionary containing all loaded datasets and derived helper
        structures. Returns None only if a truly required file is missing.
    """
    output_dir = elicitation_path / "OUTPUT"
    data_dir = elicitation_path / "DATA"
    elicitation_name = elicitation_path.name

    data = {}
    files_to_load = {
        "weights": (output_dir / f"{elicitation_name}_weights.csv", False),
        "questionnaire": (data_dir / "questionnaire.csv", True),
        "raw_seed": (data_dir / "seed.csv", False),
        "raw_target": (data_dir / "target.csv", False),
        "pc99_cooke": (output_dir / f"{elicitation_name}_pc1-99.csv", False),
        "pc99_ew": (output_dir / f"{elicitation_name}_pc1-99_EW.csv", False),
        "pc99_erf": (output_dir / f"{elicitation_name}_pc1-99_erf.csv", False),
        "samples_cooke":
        (output_dir / f"{elicitation_name}_samples.csv", False),
        "samples_ew":
        (output_dir / f"{elicitation_name}_samples_EW.csv", False),
        "samples_erf":
        (output_dir / f"{elicitation_name}_samples_erf.csv", False),
        "index_ew": (output_dir / f"{elicitation_name}_index_EW.csv", False),
        "index_cooke": (
            output_dir / f"{elicitation_name}_index_Cooke.csv",
            False,
        ),
        "index_erf": (output_dir / f"{elicitation_name}_index_ERF.csv", False),
        "val_range": (output_dir / f"{elicitation_name}_valrange.csv", False),
        "cooke_scores": (
            output_dir / f"{elicitation_name}_cooke_scores.csv",
            False,
        ),
    }

    for key, (path, is_required) in files_to_load.items():
        try:
            data[key] = pd.read_csv(path)
        except FileNotFoundError:
            if is_required:
                st.error(f"FATAL ERROR: Required file not found: {path}")
                return None
            data[key] = None

    if data.get("questionnaire") is not None:
        q_df = data["questionnaire"].copy()
        q_df.rename(columns={"SHORT_Q": "SHORT Q"},
                    inplace=True,
                    errors="ignore")

        def format_label(row):
            """Build a display label for seed and target questions."""
            q_type = row.get("QUEST_TYPE", "unknown")
            prefix = "S" if q_type == "seed" else "T"
            label = row.get("LABEL", row.get("IDX", ""))
            short_q = row.get("SHORT Q", "")
            return f"[{prefix}{label}] {short_q}"

        q_df["display_label"] = q_df.apply(format_label, axis=1)

        data["tq_df"] = q_df[q_df["QUEST_TYPE"] == "target"].reset_index(
            drop=True)
        data["sq_df"] = q_df[q_df["QUEST_TYPE"] == "seed"].reset_index(
            drop=True)
        data["all_q_df"] = pd.concat(
            [data["sq_df"], data["tq_df"]],
            ignore_index=True,
        )

        if data.get("val_range") is not None and data.get("tq_df") is not None:
            num_seed_questions = len(data["sq_df"])
            tq_ranges = data["val_range"].iloc[
                num_seed_questions:].reset_index(drop=True)
            data["tq_df"] = pd.concat(
                [
                    data["tq_df"],
                    tq_ranges[["Calculated_Min", "Calculated_Max"]],
                ],
                axis=1,
            )

        data["pie_groups"] = {}
        sum_groups = data["tq_df"][data["tq_df"]["IDXMIN"] > 0].copy()
        if not sum_groups.empty:
            sum_groups["group_id"] = (sum_groups["IDXMIN"].astype(str) + "-" +
                                      sum_groups["IDXMAX"].astype(str))
            for group_id, group_df in sum_groups.groupby("group_id"):
                min_idx, max_idx = (
                    int(group_id.split("-")[0]),
                    int(group_id.split("-")[1]),
                )
                q_indices = data["tq_df"][
                    (data["tq_df"]["IDX"] >= min_idx)
                    & (data["tq_df"]["IDX"] <= max_idx)].index.tolist()
                first_q_label = data["tq_df"].iloc[q_indices[0]]["SHORT Q"]
                group_name = f"Composition: {first_q_label}"
                data["pie_groups"][group_name] = q_indices

    combined_index_df = None
    if (index_ew_df := data.get("index_ew")) is not None:
        combined_index_df = index_ew_df.rename(
            columns={
                "Index_Mean": "EW_Mean",
                "Index_Std": "EW_Std",
                "Question_Label": "SHORT Q",
            })

    if (index_cooke_df := data.get("index_cooke")) is not None:
        cooke_df = index_cooke_df.rename(
            columns={
                "Index_Mean": "Cooke_Mean",
                "Index_Std": "Cooke_Std",
                "Question_Label": "SHORT Q",
            })
        if combined_index_df is not None:
            combined_index_df = pd.merge(
                combined_index_df,
                cooke_df,
                on="SHORT Q",
                how="outer",
            )
        else:
            combined_index_df = cooke_df

    if (index_erf_df := data.get("index_erf")) is not None:
        erf_df = index_erf_df.rename(
            columns={
                "Index_Mean": "ERF_Mean",
                "Index_Std": "ERF_Std",
                "Question_Label": "SHORT Q",
            })
        if combined_index_df is not None:
            combined_index_df = pd.merge(
                combined_index_df,
                erf_df,
                on="SHORT Q",
                how="outer",
            )
        else:
            combined_index_df = erf_df

    data["index_results"] = combined_index_df

    # Build a robust expert list, even if weights.csv is not available.
    num_experts = None
    if data.get("weights") is not None:
        num_experts = len(data["weights"])
    elif data.get("raw_target") is not None:
        num_experts = len(data["raw_target"])
    elif data.get("raw_seed") is not None:
        num_experts = len(data["raw_seed"])

    if num_experts is not None:
        expert_list = [f"Expert {i + 1}" for i in range(num_experts)]
    else:
        expert_list = []

    data["expert_list"] = expert_list

    if data.get("weights") is not None:
        data["weights"] = data["weights"].copy()
        data["weights"]["Expert"] = expert_list[:len(data["weights"])]

    if data.get("cooke_scores") is not None:
        data["cooke_scores"] = data["cooke_scores"].copy()
        data["cooke_scores"]["Expert"] = expert_list[:len(data["cooke_scores"]
                                                          )]

    return data


def get_prog_col_name(tq_index, sample_data):
    """
    Return the column name corresponding to a target question index.

    Parameters
    ----------
    tq_index : int
        Target question index.
    sample_data : pandas.DataFrame or None
        Sample dataframe used as reference.

    Returns
    -------
    str or None
        The corresponding column name if available.
    """
    if sample_data is None:
        return None
    if tq_index >= len(sample_data.columns):
        return None
    return sample_data.columns[tq_index]


def find_elicitations(base_path="."):
    """
    Find all elicitation folders containing an OUTPUT directory.

    Parameters
    ----------
    base_path : str, optional
        Base search directory.

    Returns
    -------
    dict
        Mapping of elicitation name to pathlib.Path.
    """
    elicitations_dir = Path(base_path) / "ELICITATIONS"
    if not elicitations_dir.is_dir():
        return {}

    return {
        sub_dir.name: sub_dir
        for sub_dir in elicitations_dir.iterdir()
        if sub_dir.is_dir() and (sub_dir / "OUTPUT").is_dir()
    }


def format_bin_label(value):
    """
    Format histogram bin edges compactly.

    Parameters
    ----------
    value : float
        Bin edge value.

    Returns
    -------
    str
        Formatted label.
    """
    if abs(value) > 1e-2 and abs(value) < 1e4:
        return f"{value:.2g}"
    return f"{value:.2e}"


def build_equal_weight_table(expert_list):
    """
    Build a fallback equal-weight table when weights.csv is missing.

    Parameters
    ----------
    expert_list : list[str]
        List of expert labels.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing Equal Weight percentages.
    """
    if not expert_list:
        return pd.DataFrame()

    equal_weight = 100.0 / len(expert_list)
    return pd.DataFrame({
        "Expert": expert_list,
        "Weq [%]": [equal_weight] * len(expert_list),
    })


def run():
    """Run the Streamlit dashboard."""
    st.set_page_config(layout="wide", page_title="Elicitation Dashboard")
    st.title("Interactive Elicitation Dashboard")

    elicitations = find_elicitations()
    if not elicitations:
        st.error("No valid elicitations found in 'ELICITATIONS' directory.")
        st.stop()

    selected_elicitation_name = st.sidebar.selectbox(
        "Select an Elicitation",
        list(elicitations.keys()),
    )
    elicitation_path = elicitations[selected_elicitation_name]
    data = load_all_data(elicitation_path)

    if data is None:
        st.warning(f"Could not load data for '{selected_elicitation_name}'.")
        st.stop()

    available_methods = [
        method for method in ["Cooke", "EW", "ERF"]
        if data.get(f"pc99_{method.lower()}") is not None
    ]
    if not available_methods:
        st.error(f"No result files found for '{selected_elicitation_name}'.")
        st.stop()

    sample_data_ref = data.get("samples_cooke")
    if sample_data_ref is None:
        sample_data_ref = data.get("samples_ew")
    if sample_data_ref is None:
        sample_data_ref = data.get("samples_erf")

    (
        tab_weights,
        tab_expert,
        tab_dist,
        tab_expert_drilldown,
        tab_violin,
        tab_trend,
        tab_index,
        tab_pie,
    ) = st.tabs([
        "⚖️ Weights",
        "🧑‍⚖️ Expert Answers",
        "🎯 Distributions",
        "🧑‍🔬 Expert Drill-Down",
        "🎻 Violin Plots",
        "📈 Trend Plots",
        "📉 Agreement Index",
        "📊 Pie Charts",
    ])

    with tab_dist:
        st.header("Explore Aggregated Distributions")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Controls")
            tq_display_labels = data["tq_df"]["display_label"].tolist()
            selected_display_label = st.selectbox(
                "Select a Target Question",
                tq_display_labels,
                key="dist_select",
            )
            tq_index = tq_display_labels.index(selected_display_label)
            q_info = data["tq_df"].iloc[tq_index]
            q_scale = q_info["SCALE"]
            xaxis_type = "log" if q_scale == "log" else "linear"

            methods_to_plot = st.multiselect(
                "Select Methods",
                available_methods,
                default=available_methods,
                key="dist_multi",
            )

            st.markdown("---")
            st.markdown("#### Plot Options")
            n_bins = st.slider(
                "Number of bins for histogram:",
                min_value=10,
                max_value=100,
                value=40,
                step=5,
            )

            st.markdown("---")
            st.markdown("#### Summary Statistics")
            summary_data = []
            prog_col_name = get_prog_col_name(tq_index, sample_data_ref)

            if prog_col_name:
                for method in methods_to_plot:
                    pc99_df = data.get(f"pc99_{method.lower()}")
                    samples_df = data.get(f"samples_{method.lower()}")

                    if pc99_df is not None and prog_col_name in pc99_df.columns:
                        p05 = pc99_df[prog_col_name].iloc[4]
                        p50 = pc99_df[prog_col_name].iloc[49]
                        p95 = pc99_df[prog_col_name].iloc[94]
                        mean = (samples_df[prog_col_name].mean()
                                if samples_df is not None
                                and prog_col_name in samples_df.columns else
                                "N/A")
                        summary_data.extend([
                            [f"{method} P05", p05],
                            [f"{method} P50", p50],
                            [f"{method} P95", p95],
                            [f"{method} Mean", mean],
                        ])
                st.dataframe(
                    pd.DataFrame(summary_data, columns=["Statistic", "Value"]))

        with col2:
            st.subheader("Cumulative Distribution Function (CDF)")
            fig_cdf = go.Figure()

            if prog_col_name:
                color_map = {
                    "Cooke": "#332288",
                    "EW": "#88CCEE",
                    "ERF": "#CC6677",
                }

                for method in methods_to_plot:
                    df_pc99 = data.get(f"pc99_{method.lower()}")
                    if df_pc99 is not None and prog_col_name in df_pc99.columns:
                        fig_cdf.add_trace(
                            go.Scatter(
                                x=df_pc99[prog_col_name],
                                y=df_pc99.index + 1,
                                mode="lines",
                                name=f"{method} CDF",
                                line=dict(color=color_map.get(method)),
                                hovertemplate=(
                                    "Value: %{x:.3f}<br>"
                                    "Percentile: %{y}<extra></extra>"),
                            ))

            fig_cdf.update_layout(
                title_text=f"CDF for: {selected_display_label}",
                xaxis_title="Value",
                yaxis_title="Cumulative %",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
            )
            fig_cdf.update_xaxes(type=xaxis_type)
            st.plotly_chart(fig_cdf, use_container_width=True)

            st.subheader("Probability Histogram")
            if sample_data_ref is not None and prog_col_name is not None:
                hist_data_list = []
                for method in methods_to_plot:
                    samples_df = data.get(f"samples_{method.lower()}")
                    if (samples_df is not None
                            and prog_col_name in samples_df.columns):
                        hist_data_list.append(
                            pd.DataFrame({
                                "Value": samples_df[prog_col_name],
                                "Method": method,
                            }))

                if hist_data_list:
                    hist_df = pd.concat(hist_data_list, ignore_index=True)
                    plot_df = hist_df.copy()

                    if xaxis_type == "log":
                        non_positive_count = plot_df[plot_df["Value"] <=
                                                     0].shape[0]
                        if non_positive_count > 0:
                            plot_df = plot_df[plot_df["Value"] > 0]
                            st.info(
                                f"{non_positive_count} non-positive values were "
                                "excluded from the logarithmic histogram.")

                    if not plot_df.empty:
                        min_val = plot_df["Value"].min()
                        max_val = plot_df["Value"].max()

                        if xaxis_type == "log":
                            bin_edges = np.geomspace(min_val, max_val,
                                                     n_bins + 1)
                        else:
                            bin_edges = np.linspace(min_val, max_val,
                                                    n_bins + 1)

                        bin_labels = [
                            (f"[{format_bin_label(bin_edges[i])}, "
                             f"{format_bin_label(bin_edges[i + 1])})")
                            for i in range(n_bins)
                        ]

                        plot_df["Bin"] = pd.cut(
                            plot_df["Value"],
                            bins=bin_edges,
                            labels=bin_labels,
                            right=False,
                            include_lowest=True,
                        )

                        counts = (plot_df.groupby(
                            ["Method",
                             "Bin"]).size().reset_index(name="Count"))
                        total_counts = counts.groupby(
                            "Method")["Count"].transform("sum")
                        counts["Probability"] = counts["Count"] / total_counts

                        fig_hist = px.bar(
                            counts,
                            x="Bin",
                            y="Probability",
                            color="Method",
                            barmode="overlay",
                            category_orders={"Bin": bin_labels},
                            color_discrete_map=color_map,
                            title=f"Histogram for: {selected_display_label}",
                        )
                        fig_hist.update_traces(opacity=0.9)
                        fig_hist.update_traces(
                            marker_line_width=1,
                            marker_line_color="black",
                        )
                        fig_hist.update_layout(
                            xaxis_title="Value Bins",
                            yaxis_title="Probability",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.warning(
                            "No data available to plot in the histogram after "
                            "filtering.")
            else:
                st.warning(
                    "Full sample files (`_samples.csv`) are required for "
                    "histograms.")

    with tab_weights:
        st.header("Expert Weights and Performance Metrics")
        weights_df = data.get("weights")
        cooke_scores_df = data.get("cooke_scores")
        expert_list = data.get("expert_list", [])

        if weights_df is not None:
            display_df = weights_df.copy()

            if cooke_scores_df is not None:
                score_cols = [
                    col for col in [
                        "Expert",
                        "Calibration_Score",
                        "Information_Score",
                        "unNormalized weight",
                    ] if col in cooke_scores_df.columns
                ]
                display_df = pd.merge(
                    display_df,
                    cooke_scores_df[score_cols],
                    on="Expert",
                    how="left",
                )

            rename_map = {
                "WCooke": "WCooke [%]",
                "WERF": "WERF [%]",
                "Weq": "Weq [%]",
            }
            display_df.rename(columns=rename_map, inplace=True)

            col_order = [
                "Expert",
                "Calibration_Score",
                "Information_Score",
                "unNormalized weight",
                "WCooke [%]",
                "WERF [%]",
                "Weq [%]",
            ]
            final_cols = [
                col for col in col_order if col in display_df.columns
            ]
            display_df = display_df[final_cols]

            if "Expert" in display_df.columns:
                display_df = display_df.set_index("Expert")

            format_dict = {
                "Calibration_Score": "{:.4f}",
                "Information_Score": "{:.4f}",
                "unNormalized weight": "{:.4f}",
                "WCooke [%]": "{:.2f}",
                "WERF [%]": "{:.2f}",
                "Weq [%]": "{:.2f}",
            }

            st.info("Click on column headers to sort the table.")
            st.dataframe(display_df.style.format(format_dict, na_rep="-"))

            st.subheader("Visual Comparison of Final Weights")
            plot_df = weights_df.copy()
            id_vars = ["Expert"]
            value_vars = [
                col for col in ["WCooke", "WERF", "Weq"]
                if col in plot_df.columns
            ]
            if "Expert" in plot_df.columns and value_vars:
                melted_df = plot_df.melt(
                    id_vars=id_vars,
                    value_vars=value_vars,
                    var_name="Method",
                    value_name="Weight",
                )
                fig_weights = px.bar(
                    melted_df,
                    x="Expert",
                    y="Weight",
                    color="Method",
                    barmode="group",
                    title="Normalized Weights",
                    labels={"Weight": "Weight (%)"},
                )
                st.plotly_chart(fig_weights, use_container_width=True)

        elif expert_list:
            st.info(
                "No weights file found. This elicitation appears to have been "
                "analyzed without a full weights output. Showing Equal Weight "
                "fallback values.")
            fallback_df = build_equal_weight_table(expert_list).set_index(
                "Expert")
            st.dataframe(fallback_df.style.format({"Weq [%]": "{:.2f}"}))

            fig_weights = px.bar(
                build_equal_weight_table(expert_list),
                x="Expert",
                y="Weq [%]",
                title="Equal Weights",
                labels={"Weq [%]": "Weight (%)"},
            )
            st.plotly_chart(fig_weights, use_container_width=True)
        else:
            st.warning("No expert information could be reconstructed.")

    with tab_expert:
        st.header("Analyze Individual Expert Answers")
        q_all_display_labels = data["all_q_df"]["display_label"].tolist()
        selected_q_display_label = st.selectbox(
            "Select a Question",
            q_all_display_labels,
            key="expert_select",
        )
        q_info = data["all_q_df"][data["all_q_df"]["display_label"] ==
                                  selected_q_display_label].iloc[0]

        selected_q_label = q_info["SHORT Q"]
        q_scale_expert = q_info["SCALE"]
        is_seed = q_info["QUEST_TYPE"] == "seed"
        xaxis_type_expert = "log" if q_scale_expert == "log" else "linear"

        raw_df = data.get("raw_seed") if is_seed else data.get("raw_target")

        if is_seed:
            q_idx_relative = data["sq_df"][data["sq_df"]["SHORT Q"] ==
                                           selected_q_label].index[0]
        else:
            q_idx_relative = data["tq_df"][data["tq_df"]["SHORT Q"] ==
                                           selected_q_label].index[0]

        prefix = f"{q_info['IDX']}."

        if raw_df is not None:
            relevant_cols = [
                col for col in raw_df.columns if col.startswith(prefix)
            ]
            p50_col = next((col for col in relevant_cols if "50%ile" in col),
                           None)
            p05_col = next((col for col in relevant_cols if "5%ile" in col),
                           None)
            p95_col = next((col for col in relevant_cols if "95%ile" in col),
                           None)

            if all((p05_col, p50_col, p95_col)):
                fig_ans = go.Figure()

                y_axis_all_labels = [
                    f"Expert {i + 1}" for i in range(len(raw_df))
                ]
                y_axis_positions = list(range(len(raw_df)))

                fig_ans.add_trace(
                    go.Scatter(
                        y=y_axis_positions,
                        x=raw_df[p50_col],
                        error_x=dict(
                            type="data",
                            symmetric=False,
                            array=raw_df[p95_col] - raw_df[p50_col],
                            arrayminus=raw_df[p50_col] - raw_df[p05_col],
                        ),
                        mode="markers",
                        marker=dict(color="blue"),
                        name="Experts",
                    ))

                if not is_seed:
                    prog_col_name = get_prog_col_name(q_idx_relative,
                                                      sample_data_ref)
                    if prog_col_name:
                        dm_pos_counter = len(raw_df)
                        for method, color in [
                            ("Cooke", "#332288"),
                            ("EW", "#88CCEE"),
                            ("ERF", "#CC6677"),
                        ]:
                            pc99_df = data.get(f"pc99_{method.lower()}")
                            if (pc99_df is not None
                                    and prog_col_name in pc99_df.columns):
                                p50_dm = pc99_df[prog_col_name].iloc[49]
                                p05_dm = pc99_df[prog_col_name].iloc[4]
                                p95_dm = pc99_df[prog_col_name].iloc[94]
                                fig_ans.add_trace(
                                    go.Scatter(
                                        y=[dm_pos_counter],
                                        x=[p50_dm],
                                        error_x=dict(
                                            type="data",
                                            symmetric=False,
                                            array=[p95_dm - p50_dm],
                                            arrayminus=[p50_dm - p05_dm],
                                        ),
                                        mode="markers",
                                        marker=dict(
                                            color=color,
                                            size=12,
                                            symbol="diamond",
                                        ),
                                        name=f"DM ({method})",
                                    ))
                                y_axis_all_labels.append(f"DM - {method}")
                                y_axis_positions.append(dm_pos_counter)
                                dm_pos_counter += 1

                if is_seed:
                    realization_pos = len(raw_df)
                    fig_ans.add_trace(
                        go.Scatter(
                            y=[realization_pos],
                            x=[q_info["REALIZATION"]],
                            mode="markers",
                            marker=dict(color="black", size=14, symbol="x"),
                            name="Realization",
                        ))
                    y_axis_all_labels.append("Realization")
                    y_axis_positions.append(realization_pos)

                fig_ans.update_layout(
                    title=f"Expert Answers for: {selected_q_display_label}",
                    yaxis_title="Source",
                    xaxis_title="Value",
                    xaxis_type=xaxis_type_expert,
                    yaxis=dict(
                        tickmode="array",
                        tickvals=y_axis_positions,
                        ticktext=y_axis_all_labels,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="LightGray",
                    ),
                    yaxis_autorange="reversed",
                )
                fig_ans.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="LightGray",
                )
                st.plotly_chart(fig_ans, use_container_width=True)
            else:
                st.warning(
                    f"Could not find data columns for '{selected_q_label}'.")

    with tab_expert_drilldown:
        st.header("Single Expert Answer Analysis")
        expert_list = data.get("expert_list", [])

        if expert_list:
            col1, col2 = st.columns([1, 3])

            with col1:
                selected_expert = st.selectbox(
                    "Select an Expert",
                    expert_list,
                    key="expert_drill_select",
                )
                st.markdown("---")
                st.markdown("#### Comparison Options")
                available_aggr = [
                    f"{method} {percentile}" for method in available_methods
                    for percentile in ["P05", "P50", "P95"]
                ]
                selected_aggr = st.multiselect(
                    "Select DM percentiles to show:",
                    options=available_aggr,
                    default=[f"{method} P50" for method in available_methods],
                )

            with col2:
                st.subheader(f"Answers for {selected_expert}")
                expert_index = expert_list.index(selected_expert)
                seed_answers = data.get("raw_seed")
                target_answers = data.get("raw_target")
                expert_responses = []

                if seed_answers is not None and expert_index < len(
                        seed_answers):
                    expert_row_seed = seed_answers.iloc[expert_index]
                    for _, q_info in data["sq_df"].iterrows():
                        prefix = f"{q_info['IDX']}."
                        p05_col = next(
                            (col for col in seed_answers.columns
                             if col.startswith(prefix) and "5%ile" in col),
                            None,
                        )
                        p50_col = next(
                            (col for col in seed_answers.columns
                             if col.startswith(prefix) and "50%ile" in col),
                            None,
                        )
                        p95_col = next(
                            (col for col in seed_answers.columns
                             if col.startswith(prefix) and "95%ile" in col),
                            None,
                        )

                        if all((p05_col, p50_col, p95_col)):
                            expert_responses.append({
                                "Question":
                                q_info["display_label"],
                                "Type":
                                "Seed",
                                "Expert P05":
                                expert_row_seed[p05_col],
                                "Expert P50":
                                expert_row_seed[p50_col],
                                "Expert P95":
                                expert_row_seed[p95_col],
                                "Realization":
                                q_info["REALIZATION"],
                            })

                if target_answers is not None and expert_index < len(
                        target_answers):
                    expert_row_target = target_answers.iloc[expert_index]
                    for i, q_info in data["tq_df"].iterrows():
                        prefix = f"{q_info['IDX']}."
                        prog_col_name = get_prog_col_name(i, sample_data_ref)
                        p05_col = next(
                            (col for col in target_answers.columns
                             if col.startswith(prefix) and "5%ile" in col),
                            None,
                        )
                        p50_col = next(
                            (col for col in target_answers.columns
                             if col.startswith(prefix) and "50%ile" in col),
                            None,
                        )
                        p95_col = next(
                            (col for col in target_answers.columns
                             if col.startswith(prefix) and "95%ile" in col),
                            None,
                        )

                        if all((p05_col, p50_col, p95_col)):
                            response_dict = {
                                "Question": q_info["display_label"],
                                "Type": "Target",
                                "Expert P05": expert_row_target[p05_col],
                                "Expert P50": expert_row_target[p50_col],
                                "Expert P95": expert_row_target[p95_col],
                                "Realization": None,
                            }

                            for aggr in selected_aggr:
                                method, percentile = aggr.split(" ")
                                pc_map = {"P05": 4, "P50": 49, "P95": 94}
                                pc_index = pc_map.get(percentile)
                                pc99_df = data.get(f"pc99_{method.lower()}")

                                if (pc99_df is not None
                                        and prog_col_name in pc99_df.columns
                                        and pc_index is not None):
                                    response_dict[aggr] = pc99_df[
                                        prog_col_name].iloc[pc_index]

                            expert_responses.append(response_dict)

                if expert_responses:
                    df_display = pd.DataFrame(expert_responses)

                    if "Question" in df_display.columns:
                        df_display = df_display.set_index("Question")

                    format_dict = {
                        col: "{:,.2f}"
                        for col in df_display.columns
                        if pd.api.types.is_numeric_dtype(df_display[col])
                    }
                    st.dataframe(
                        df_display.style.format(format_dict, na_rep="-"))
        else:
            st.warning("No expert list could be reconstructed from the data.")

    with tab_violin:
        st.header("Violin Plots for Selected Questions")
        if sample_data_ref is not None:
            tq_df = data["tq_df"]
            all_display_labels = tq_df["display_label"].tolist()

            if "violin_selection" not in st.session_state:
                st.session_state.violin_selection = all_display_labels[:3]

            options = all_display_labels
            if st.session_state.violin_selection:
                try:
                    first_selected_info = tq_df[
                        tq_df["display_label"] ==
                        st.session_state.violin_selection[0]].iloc[0]
                    compatible_mask = (
                        tq_df["UNITS"] == first_selected_info["UNITS"]) & (
                            tq_df["SCALE"] == first_selected_info["SCALE"])
                    options = tq_df[compatible_mask]["display_label"].tolist()
                except IndexError:
                    options = all_display_labels

            st.session_state.violin_selection = [
                selection for selection in st.session_state.violin_selection
                if selection in options
            ]

            if st.button("Clear Violin Selections"):
                st.session_state.violin_selection = []
                st.rerun()

            selected_display_labels = st.multiselect(
                "Select questions to compare:",
                options=options,
                key="violin_selection",
            )

            if selected_display_labels:
                tq_indices = [
                    all_display_labels.index(label)
                    for label in selected_display_labels
                ]
                anchor_question_info = tq_df.iloc[tq_indices[0]]
                scale_type = ("log" if anchor_question_info["SCALE"] == "log"
                              else "linear")

                cols_to_plot = [
                    get_prog_col_name(i, sample_data_ref) for i in tq_indices
                ]
                descriptive_names = [
                    tq_df["SHORT Q"].iloc[i] for i in tq_indices
                ]

                methods = st.multiselect(
                    "Select Methods",
                    available_methods,
                    default=available_methods,
                    key="violin_multi_select",
                )

                all_samples = []
                for method in methods:
                    df_samples = data.get(f"samples_{method.lower()}")
                    if df_samples is not None:
                        valid_cols = [
                            col for col in cols_to_plot
                            if col in df_samples.columns
                        ]
                        subset = df_samples[valid_cols]
                        if not subset.empty:
                            subset = subset.copy()
                            subset.columns = descriptive_names[:len(subset.
                                                                    columns)]
                            melted = subset.melt(
                                var_name="Question",
                                value_name="Value",
                            ).assign(Method=method)
                            all_samples.append(melted)

                if all_samples:
                    combined_df = pd.concat(all_samples, ignore_index=True)
                    y_axis_title = f"Value ({anchor_question_info['UNITS']})"
                    plot_column = "Value"

                    if scale_type == "log":
                        combined_df["Value_log10"] = np.log10(
                            combined_df["Value"])
                        plot_column = "Value_log10"
                        y_axis_title = (
                            f"Value ({anchor_question_info['UNITS']}) [log10 scale]"
                        )

                    fig = px.violin(
                        combined_df,
                        x="Question",
                        y=plot_column,
                        color="Method",
                        box=True,
                        points=False,
                        title="Distribution Comparison for Selected Questions",
                    )
                    fig.update_layout(yaxis_title=y_axis_title)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                "Sample files (`_samples.csv`) are required for Violin Plots.")

    with tab_trend:
        st.header("Trend Plots for Selected Questions")
        tq_df = data["tq_df"]
        all_display_labels = tq_df["display_label"].tolist()

        if "trend_selection" not in st.session_state:
            st.session_state.trend_selection = all_display_labels[:5]

        options_trend = all_display_labels
        if st.session_state.trend_selection:
            try:
                first_selected_info = tq_df[
                    tq_df["display_label"] ==
                    st.session_state.trend_selection[0]].iloc[0]
                compatible_mask = (
                    tq_df["UNITS"] == first_selected_info["UNITS"]) & (
                        tq_df["SCALE"] == first_selected_info["SCALE"])
                options_trend = tq_df[compatible_mask]["display_label"].tolist(
                )
            except IndexError:
                options_trend = all_display_labels

        st.session_state.trend_selection = [
            selection for selection in st.session_state.trend_selection
            if selection in options_trend
        ]

        if st.button("Clear Trend Selections"):
            st.session_state.trend_selection = []
            st.rerun()

        selected_display_labels = st.multiselect(
            "Select questions for Trend Plot:",
            options=options_trend,
            key="trend_selection",
        )

        if selected_display_labels:
            tq_indices = [
                all_display_labels.index(label)
                for label in selected_display_labels
            ]
            anchor_question_info = tq_df.iloc[tq_indices[0]]
            yaxis_type_trend = ("log" if anchor_question_info["SCALE"] == "log"
                                else "linear")

            min_val = anchor_question_info["MINVAL"]
            max_val = anchor_question_info["MAXVAL"]

            y_range = [
                min_val if np.isfinite(min_val) else None,
                max_val if np.isfinite(max_val) else None,
            ]

            if yaxis_type_trend == "log" and y_range[
                    0] is not None and y_range[0] <= 0:
                y_range[0] = None

            descriptive_names = [tq_df["SHORT Q"].iloc[i] for i in tq_indices]
            prog_col_names = [
                get_prog_col_name(i, sample_data_ref) for i in tq_indices
            ]

            fig_trend = go.Figure()
            num_methods = len(available_methods)
            offsets = (np.linspace(-0.2, 0.2, num_methods)
                       if num_methods > 1 else [0])

            for i, method in enumerate(available_methods):
                pc99_df = data.get(f"pc99_{method.lower()}")
                if pc99_df is not None:
                    valid_prog_cols = [
                        col for col in prog_col_names if col in pc99_df.columns
                    ]
                    if valid_prog_cols:
                        p50s = [
                            pc99_df[col].iloc[49] for col in valid_prog_cols
                        ]
                        p05s = [
                            pc99_df[col].iloc[4] for col in valid_prog_cols
                        ]
                        p95s = [
                            pc99_df[col].iloc[94] for col in valid_prog_cols
                        ]

                        x_data = [j + offsets[i] for j in range(len(p50s))]
                        fig_trend.add_trace(
                            go.Scatter(
                                x=x_data,
                                y=p50s,
                                error_y=dict(
                                    type="data",
                                    symmetric=False,
                                    array=np.array(p95s) - np.array(p50s),
                                    arrayminus=np.array(p50s) - np.array(p05s),
                                ),
                                mode="markers",
                                name=method,
                            ))

            fig_trend.update_layout(
                title="Trend Plot for Selected Questions",
                yaxis_title=f"Value ({anchor_question_info['UNITS']})",
                yaxis_type=yaxis_type_trend,
                yaxis_range=y_range,
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(descriptive_names))),
                    ticktext=descriptive_names,
                ),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    with tab_index:
        st.header("Agreement Index Analysis")
        index_df = data.get("index_results")

        if index_df is not None:
            index_df_merged = pd.merge(
                index_df,
                data["tq_df"][["SHORT Q", "display_label"]],
                on="SHORT Q",
                how="left",
            )
            all_display_labels = index_df_merged["display_label"].dropna(
            ).tolist()
            selected_labels = st.multiselect(
                "Select questions:",
                options=all_display_labels,
                default=all_display_labels[:5],
                key="index_select",
            )

            if selected_labels:
                sub_df = index_df_merged[index_df_merged["display_label"].isin(
                    selected_labels)]
                fig = go.Figure()
                methods = [
                    method for method in ["Cooke", "EW", "ERF"]
                    if f"{method}_Mean" in sub_df.columns
                ]
                offsets = (np.linspace(-0.2, 0.2, len(methods))
                           if len(methods) > 1 else [0])
                y_map = {label: i for i, label in enumerate(selected_labels)}

                for i, method in enumerate(methods):
                    mean_col = f"{method}_Mean"
                    std_col = f"{method}_Std"
                    if mean_col in sub_df.columns and std_col in sub_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=sub_df[mean_col],
                                y=[
                                    y_map.get(label) + offsets[i]
                                    for label in sub_df["display_label"]
                                ],
                                error_x=dict(type="data",
                                             array=sub_df[std_col]),
                                mode="markers",
                                name=method,
                                marker=dict(size=8),
                            ))

                fig.update_layout(
                    xaxis_range=[-1, 1],
                    yaxis=dict(
                        tickmode="array",
                        tickvals=list(range(len(selected_labels))),
                        ticktext=[
                            label.split("] ")[1] for label in selected_labels
                        ],
                    ),
                    height=max(400,
                               len(selected_labels) * 50),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No index result files were found.")

    with tab_pie:
        st.header("Compositional Group Pie Charts")
        pie_groups = data.get("pie_groups", {})

        if sample_data_ref is not None and pie_groups:
            selected_group = st.selectbox(
                "Select a Group",
                list(pie_groups.keys()),
                key="pie_select",
            )
            tq_df = data["tq_df"]
            indices = pie_groups[selected_group]
            names = [tq_df["SHORT Q"].iloc[i] for i in indices]
            prog_cols = [
                get_prog_col_name(i, sample_data_ref) for i in indices
            ]
            target_sum = tq_df.iloc[indices[0]]["SUM50"]

            cols = st.columns(len(available_methods))
            for i, method in enumerate(available_methods):
                with cols[i]:
                    st.subheader(f"{method} Method")
                    samples = data.get(f"samples_{method.lower()}")
                    if samples is not None:
                        valid_cols = [
                            col for col in prog_cols if col in samples.columns
                        ]
                        if valid_cols:
                            means = samples[valid_cols].mean().values
                            if means.sum() > 0:
                                norm_means = (means / means.sum()) * target_sum
                            else:
                                norm_means = means

                            pie_data = pd.DataFrame({
                                "Question": names,
                                "Mean Value": norm_means,
                            })
                            fig_pie = px.pie(
                                pie_data,
                                values="Mean Value",
                                names="Question",
                                title=f"Proportions (Sum={target_sum})",
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Sample files or compositional groups are required.")


if __name__ == "__main__":
    run()
