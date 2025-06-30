import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# --- Configuration ---
PIE_GROUPS = {"Andesitic Eruption Style (Next Eruption)": [5, 6, 7, 8, 9]}


@st.cache_data
def load_all_data(elicitation_path):
    """Loads all necessary CSV files for a given elicitation path."""
    output_dir = elicitation_path / "OUTPUT"
    data_dir = elicitation_path / "DATA"
    elicitation_name = elicitation_path.name

    data = {}
    files_to_load = {
        "weights": (output_dir / f"{elicitation_name}_weights.csv", True),
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
        "index_cooke":
        (output_dir / f"{elicitation_name}_index_Cooke.csv", False),
        "index_erf": (output_dir / f"{elicitation_name}_index_ERF.csv", False),
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
        q_df = data["questionnaire"]
        q_df.rename(columns={"SHORT_Q": "SHORT Q"},
                    inplace=True,
                    errors='ignore')

        def format_label(row):
            label, short_q = row.get('LABEL',
                                     row.get('IDX',
                                             '')), row.get('SHORT Q', '')
            return f"[{label}] {short_q}"

        q_df["display_label"] = q_df.apply(format_label, axis=1)

        data["tq_df"] = q_df[q_df["QUEST_TYPE"] == 'target'].reset_index(
            drop=True)
        data["sq_df"] = q_df[q_df["QUEST_TYPE"] == 'seed'].reset_index(
            drop=True)
        data['all_q_df'] = pd.concat([data["sq_df"], data["tq_df"]],
                                     ignore_index=True)

        data['pie_groups'] = {}
        sum_groups = data['tq_df'][data['tq_df']['IDXMIN'] > 0].copy()
        if not sum_groups.empty:
            sum_groups['group_id'] = sum_groups['IDXMIN'].astype(
                str) + '-' + sum_groups['IDXMAX'].astype(str)
            for group_id, group_df in sum_groups.groupby('group_id'):
                min_idx, max_idx = int(group_id.split('-')[0]), int(
                    group_id.split('-')[1])
                q_indices = data['tq_df'][(data['tq_df']['IDX'] >= min_idx) & (
                    data['tq_df']['IDX'] <= max_idx)].index.tolist()
                first_q_label = data['tq_df'].iloc[q_indices[0]]['SHORT Q']
                group_name = f"Composition: {first_q_label}"
                data['pie_groups'][group_name] = q_indices

    combined_index_df = None
    if (index_ew_df := data.get("index_ew")) is not None:
        combined_index_df = index_ew_df.rename(
            columns={
                "Index_Mean": "EW_Mean",
                "Index_Std": "EW_Std",
                "Question_Label": "SHORT Q"
            })
    if (index_cooke_df := data.get("index_cooke")) is not None:
        cooke_df = index_cooke_df.rename(
            columns={
                "Index_Mean": "Cooke_Mean",
                "Index_Std": "Cooke_Std",
                "Question_Label": "SHORT Q"
            })
        if combined_index_df is not None:
            combined_index_df = pd.merge(combined_index_df,
                                         cooke_df,
                                         on="SHORT Q",
                                         how="outer")
        else:
            combined_index_df = cooke_df
    data["index_results"] = combined_index_df

    if data.get("weights") is not None:
        data["weights"]["Expert"] = [
            f"Expert {i+1}" for i in range(len(data["weights"]))
        ]
    return data


def get_prog_col_name(tq_index, sample_data):
    return sample_data.columns[
        tq_index] if sample_data is not None and tq_index < len(
            sample_data.columns) else None


def find_elicitations(base_path="."):
    elicitations_dir = Path(base_path) / "ELICITATIONS"
    if not elicitations_dir.is_dir():
        return {}
    return {
        sub_dir.name: sub_dir
        for sub_dir in elicitations_dir.iterdir()
        if sub_dir.is_dir() and (sub_dir / "OUTPUT").is_dir()
    }


def run():
    st.set_page_config(layout="wide", page_title="Elicitation Dashboard")
    st.title("Interactive Elicitation Dashboard")

    elicitations = find_elicitations()
    if not elicitations:
        st.error("No valid elicitations found in 'ELICITATIONS' directory.")
        st.stop()

    selected_elicitation_name = st.sidebar.selectbox("Select an Elicitation",
                                                     list(elicitations.keys()))
    elicitation_path = elicitations[selected_elicitation_name]
    data = load_all_data(elicitation_path)

    if data is None:
        st.warning(f"Could not load data for '{selected_elicitation_name}'.")
        st.stop()

    available_methods = [
        m for m in ["Cooke", "EW", "ERF"]
        if data.get(f"pc99_{m.lower()}") is not None
    ]
    if not available_methods:
        st.error(f"No result files found for '{selected_elicitation_name}'.")
        st.stop()

    # --- CORRECTED LOGIC ---
    sample_data_ref = data.get("samples_cooke")
    if sample_data_ref is None:
        sample_data_ref = data.get("samples_ew")

    tab_weights, tab_expert, tab_dist, tab_violin, tab_trend, tab_index, tab_pie = st.tabs(
        [
            "⚖️ Weights", "🧑‍⚖️ Expert Answers", "🎯 Distributions",
            "🎻 Violin Plots", "📈 Trend Plots", "📉 Agreement Index",
            "📊 Pie Charts"
        ])

    with tab_dist:
        st.header("Explore Aggregated Distributions")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Controls")
            tq_display_labels = data["tq_df"]["display_label"].tolist()
            selected_display_label = st.selectbox("Select a Target Question", tq_display_labels, key="dist_select")
            tq_index = tq_display_labels.index(selected_display_label)
            selected_tq_label = data["tq_df"].iloc[tq_index]["SHORT Q"]
            q_scale = data["tq_df"].iloc[tq_index]['SCALE']
            xaxis_type = 'log' if q_scale == 'log' else 'linear'
            methods_to_plot = st.multiselect("Select Methods", available_methods, default=available_methods, key="dist_multi")
            
            st.markdown("---")
            st.markdown("#### Plot Options")
            n_bins = st.slider("Number of bins for histogram:", min_value=10, max_value=200, value=50, step=10)
            
            st.markdown("---")
            st.markdown("#### Summary Statistics")
            summary_data, prog_col_name = [], get_prog_col_name(tq_index, sample_data_ref)
            if prog_col_name:
                for method in methods_to_plot:
                    pc99_df, samples_df = data.get(f"pc99_{method.lower()}"), data.get(f"samples_{method.lower()}")
                    if pc99_df is not None:
                        p05, p50, p95 = pc99_df[prog_col_name].iloc[4], pc99_df[prog_col_name].iloc[49], pc99_df[prog_col_name].iloc[94]
                        mean = samples_df[prog_col_name].mean() if samples_df is not None else "N/A"
                        summary_data.extend([[f"{method} P05", p05], [f"{method} P50", p50], [f"{method} P95", p95], [f"{method} Mean", mean]])
                st.dataframe(pd.DataFrame(summary_data, columns=["Statistic", "Value"]))
        with col2:
            st.subheader("Cumulative Distribution Function (CDF)")
            fig_cdf = go.Figure()
            if prog_col_name:
                for method in methods_to_plot:
                    df_pc99 = data.get(f"pc99_{method.lower()}")
                    if df_pc99 is not None and prog_col_name in df_pc99.columns:
                        fig_cdf.add_trace(go.Scatter(x=df_pc99[prog_col_name], y=df_pc99.index + 1, mode='lines', name=f"{method} CDF", hovertemplate="Value: %{x:.3f}<br>Percentile: %{y}<extra></extra>"))
            fig_cdf.update_layout(title_text=f"CDF for: {selected_display_label}", xaxis_title="Value", yaxis_title="Cumulative %", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_cdf.update_xaxes(type=xaxis_type)
            st.plotly_chart(fig_cdf, use_container_width=True)
            
            st.subheader("Probability Density Histogram")
            if sample_data_ref is not None:
                hist_data_list = [pd.DataFrame({'Value': data.get(f"samples_{m.lower()}")[prog_col_name], 'Method': m}) for m in methods_to_plot if data.get(f"samples_{m.lower()}") is not None and prog_col_name in data.get(f"samples_{m.lower()}").columns]
                if hist_data_list:
                    hist_df = pd.concat(hist_data_list)
                    fig_hist = px.histogram(hist_df, x="Value", color="Method", barmode="overlay", histnorm='probability density', log_x=(xaxis_type == 'log'), nbins=n_bins, title=f"Histogram for: {selected_display_label}")
                    fig_hist.update_traces(opacity=0.4)
                    st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("Full sample files (`_samples.csv`) are required for histograms.")

    with tab_weights:
        st.header("Expert Weights and Performance Metrics")
        weights_df = data.get("weights")
        if weights_df is not None:
            display_df = weights_df.copy()
            for col in ['WCooke', 'WERF', 'Weq']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map('{:.2f}%'.format)
            if 'Expert' in display_df.columns:
                display_df = display_df.set_index('Expert')
            if 'index' in display_df.columns:
                display_df = display_df.drop(columns=['index'])
            st.dataframe(display_df)
            st.subheader("Visual Comparison of Expert Weights")
            plot_df, id_vars, value_vars = weights_df.copy(), ['Expert'], [
                c for c in ['WCooke', 'WERF', 'Weq'] if c in weights_df.columns
            ]
            if id_vars[0] in plot_df.columns and value_vars:
                melted_df = plot_df.melt(id_vars=id_vars,
                                         value_vars=value_vars,
                                         var_name='Method',
                                         value_name='Weight')
                fig_weights = px.bar(melted_df,
                                     x='Expert',
                                     y='Weight',
                                     color='Method',
                                     barmode='group',
                                     title='Normalized Weights',
                                     labels={'Weight': 'Weight (%)'})
                st.plotly_chart(fig_weights, use_container_width=True)
        else:
            st.warning("Weights file not found.")

    with tab_expert:
        st.header("Analyze Individual Expert Answers")
        q_all_display_labels = data["all_q_df"]["display_label"].tolist()
        selected_q_display_label = st.selectbox("Select a Question",
                                                q_all_display_labels,
                                                key="expert_select")
        q_info = data["all_q_df"][data["all_q_df"]["display_label"] ==
                                  selected_q_display_label].iloc[0]
        selected_q_label, q_scale_expert, is_seed = q_info["SHORT Q"], q_info[
            'SCALE'], q_info["QUEST_TYPE"] == "seed"
        xaxis_type_expert = 'log' if q_scale_expert == 'log' else 'linear'
        raw_df = data.get("raw_seed") if is_seed else data.get("raw_target")
        q_idx_relative = data["sq_df"][
            data["sq_df"]["SHORT Q"] ==
            selected_q_label].index[0] if is_seed else data["tq_df"][
                data["tq_df"]["SHORT Q"] == selected_q_label].index[0]
        prefix = f"{q_info['IDX']}."
        if raw_df is not None:
            relevant_cols = [c for c in raw_df.columns if c.startswith(prefix)]
            p50, p05, p95 = next(
                (c for c in relevant_cols if "50%ile" in c), None), next(
                    (c for c in relevant_cols if "5%ile" in c), None), next(
                        (c for c in relevant_cols if "95%ile" in c), None)
            if all((p05, p50, p95)):
                fig_ans = go.Figure()
                expert_y_labels, y_axis_all_labels, y_axis_positions = [
                    f"Expert {i+1}" for i in range(len(raw_df))
                ], [f"Expert {i+1}"
                    for i in range(len(raw_df))], list(range(len(raw_df)))
                fig_ans.add_trace(
                    go.Scatter(y=y_axis_positions,
                               x=raw_df[p50],
                               error_x=dict(type='data',
                                            symmetric=False,
                                            array=raw_df[p95] - raw_df[p50],
                                            arrayminus=raw_df[p50] -
                                            raw_df[p05]),
                               mode='markers',
                               marker=dict(color='blue'),
                               name='Experts'))
                if not is_seed:
                    prog_col_name = get_prog_col_name(q_idx_relative,
                                                      sample_data_ref)
                    if prog_col_name:
                        dm_pos_counter = len(raw_df)
                        for method, color in [("Cooke", "firebrick"),
                                              ("EW", "forestgreen"),
                                              ("ERF", "royalblue")]:
                            pc99_df = data.get(f"pc99_{method.lower()}")
                            if pc99_df is not None:
                                p50_dm, p05_dm, p95_dm = pc99_df[
                                    prog_col_name].iloc[49], pc99_df[
                                        prog_col_name].iloc[4], pc99_df[
                                            prog_col_name].iloc[94]
                                fig_ans.add_trace(
                                    go.Scatter(
                                        y=[dm_pos_counter],
                                        x=[p50_dm],
                                        error_x=dict(
                                            type='data',
                                            symmetric=False,
                                            array=[p95_dm - p50_dm],
                                            arrayminus=[p50_dm - p05_dm]),
                                        mode='markers',
                                        marker=dict(color=color,
                                                    size=12,
                                                    symbol='diamond'),
                                        name=f"DM ({method})"))
                                y_axis_all_labels.append(f"DM - {method}")
                                y_axis_positions.append(dm_pos_counter)
                                dm_pos_counter += 1
                if is_seed:
                    realization_pos = len(raw_df)
                    fig_ans.add_trace(
                        go.Scatter(y=[realization_pos],
                                   x=[q_info['REALIZATION']],
                                   mode='markers',
                                   marker=dict(color='black',
                                               size=14,
                                               symbol='x'),
                                   name='Realization'))
                    y_axis_all_labels.append("Realization")
                    y_axis_positions.append(realization_pos)
                fig_ans.update_layout(
                    title=f'Expert Answers for: {selected_q_display_label}',
                    yaxis_title="Source",
                    xaxis_title="Value",
                    xaxis_type=xaxis_type_expert,
                    yaxis=dict(tickmode='array',
                               tickvals=y_axis_positions,
                               ticktext=y_axis_all_labels,
                               showgrid=True,
                               gridwidth=1,
                               gridcolor='LightGray'),
                    yaxis_autorange='reversed')
                fig_ans.update_xaxes(showgrid=True,
                                     gridwidth=1,
                                     gridcolor='LightGray')
                st.plotly_chart(fig_ans, use_container_width=True)
            else:
                st.warning(
                    f"Could not find data columns for '{selected_q_label}'.")

    with tab_violin:
        st.header("Violin Plots for Selected Questions")
        if sample_data_ref is not None:
            tq_df = data["tq_df"]
            all_display_labels = tq_df["display_label"].tolist()
            if 'violin_selection' not in st.session_state:
                st.session_state.violin_selection = all_display_labels[:3]
            options = all_display_labels
            if st.session_state.violin_selection:
                try:
                    first_info = tq_df[
                        tq_df["display_label"] ==
                        st.session_state.violin_selection[0]].iloc[0]
                    mask = (tq_df['UNITS'] == first_info['UNITS']) & (
                        tq_df['SCALE'] == first_info['SCALE'])
                    options = tq_df[mask]["display_label"].tolist()
                except IndexError:
                    options = all_display_labels
            st.session_state.violin_selection = [
                s for s in st.session_state.violin_selection if s in options
            ]
            if st.button("Clear Violin Selections"):
                st.session_state.violin_selection = []
                st.rerun()
            selected_labels = st.multiselect("Select questions:",
                                             options=options,
                                             key="violin_selection")
            if selected_labels:
                indices = [
                    all_display_labels.index(l) for l in selected_labels
                ]
                info = tq_df.iloc[indices[0]]
                y_type = 'log' if info['SCALE'] == 'log' else 'linear'
                cols, names = [
                    get_prog_col_name(i, sample_data_ref) for i in indices
                ], [tq_df["SHORT Q"].iloc[i] for i in indices]
                methods = st.multiselect("Select Methods",
                                         available_methods,
                                         default=available_methods,
                                         key="violin_multi")
                samples = [
                    df for m in methods
                    if (s_df := data.get(f"samples_{m.lower()}")) is not None
                    for df in [
                        pd.melt(s_df[[
                            c for c in cols if c in s_df.columns
                        ]].rename(columns=dict(
                            zip([c for c in cols
                                 if c in s_df.columns], names[:len(
                                     [c for c in cols
                                      if c in s_df.columns])]))),
                                var_name="Question",
                                value_name="Value").assign(Method=m)
                    ]
                ]
                if samples:
                    df = pd.concat(samples)
                    min_v, max_v = df['Value'].min(), df['Value'].max()
                    y_range = [
                        np.log10(df['Value'][df['Value'] > 0].min())
                        if y_type == 'log'
                        and df['Value'][df['Value'] > 0].any() else min_v,
                        np.log10(max_v) if y_type == 'log' else max_v
                    ]
                    fig = px.violin(df,
                                    x="Question",
                                    y="Value",
                                    color="Method",
                                    box=True,
                                    points=False)
                    fig.update_yaxes(type=y_type, range=y_range)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sample files are required.")

    with tab_trend:
        st.header("Trend Plots for Selected Questions")
        tq_df, all_display_labels = data["tq_df"], data["tq_df"][
            "display_label"].tolist()
        if 'trend_selection' not in st.session_state:
            st.session_state.trend_selection = all_display_labels[:5]
        options = all_display_labels
        if st.session_state.trend_selection:
            try:
                first_info = tq_df[tq_df["display_label"] ==
                                   st.session_state.trend_selection[0]].iloc[0]
                mask = (tq_df['UNITS'] == first_info['UNITS']) & (
                    tq_df['SCALE'] == first_info['SCALE'])
                options = tq_df[mask]["display_label"].tolist()
            except IndexError:
                options = all_display_labels
        st.session_state.trend_selection = [
            s for s in st.session_state.trend_selection if s in options
        ]
        if st.button("Clear Trend Selections"):
            st.session_state.trend_selection = []
            st.rerun()
        selected_labels = st.multiselect("Select questions:",
                                         options=options,
                                         key="trend_selection")
        if selected_labels:
            indices = [all_display_labels.index(l) for l in selected_labels]
            info, y_type = tq_df.iloc[indices[0]], 'log' if tq_df.iloc[
                indices[0]]['SCALE'] == 'log' else 'linear'
            all_p05, all_p95 = [], []
            for m in available_methods:
                if (pc99 := data.get(f"pc99_{m.lower()}")) is not None:
                    prog_cols = [
                        get_prog_col_name(i, sample_data_ref) for i in indices
                    ]
                    all_p05.extend(
                        [pc99[c].iloc[4] for c in prog_cols if c in pc99])
                    all_p95.extend(
                        [pc99[c].iloc[94] for c in prog_cols if c in pc99])
            y_range = [
                min(all_p05) if all_p05 else None,
                max(all_p95) if all_p95 else None
            ]
            if y_type == 'log' and y_range[0] is not None and y_range[0] <= 0:
                y_range[0] = None
            names, prog_cols = [tq_df["SHORT Q"].iloc[i] for i in indices], [
                get_prog_col_name(i, sample_data_ref) for i in indices
            ]
            fig = go.Figure()
            offsets = np.linspace(
                -0.2, 0.2,
                len(available_methods)) if len(available_methods) > 1 else [0]
            for i, m in enumerate(available_methods):
                if (pc99 := data.get(f"pc99_{m.lower()}")) is not None:
                    p50s, p05s, p95s = [
                        pc99[c].iloc[49] for c in prog_cols if c in pc99
                    ], [pc99[c].iloc[4] for c in prog_cols if c in pc99
                        ], [pc99[c].iloc[94] for c in prog_cols if c in pc99]
                    if p50s:
                        fig.add_trace(
                            go.Scatter(
                                x=[j + offsets[i] for j in range(len(names))],
                                y=p50s,
                                error_y=dict(type='data',
                                             symmetric=False,
                                             array=np.array(p95s) -
                                             np.array(p50s),
                                             arrayminus=np.array(p50s) -
                                             np.array(p05s)),
                                mode='markers',
                                name=m))
            fig.update_layout(yaxis_title=f"Value ({info['UNITS']})",
                              yaxis_type=y_type,
                              yaxis_range=y_range,
                              xaxis=dict(tickmode='array',
                                         tickvals=list(range(len(names))),
                                         ticktext=names))
            st.plotly_chart(fig, use_container_width=True)

    with tab_index:
        st.header("Agreement Index Analysis")
        index_df = data.get("index_results")
        if index_df is not None:
            index_df_merged = pd.merge(
                index_df,
                data["tq_df"][['SHORT Q', 'display_label']],
                on='SHORT Q',
                how='left')
            all_display_labels = index_df_merged["display_label"].dropna(
            ).tolist()
            selected_labels = st.multiselect("Select questions:",
                                             options=all_display_labels,
                                             default=all_display_labels[:5],
                                             key="index_select")
            if selected_labels:
                sub_df = index_df_merged[index_df_merged["display_label"].isin(
                    selected_labels)]
                fig = go.Figure()
                methods = [
                    m for m in ["Cooke", "EW", "ERF"]
                    if f"{m}_Mean" in sub_df.columns
                ]
                offsets = np.linspace(
                    -0.2, 0.2, len(methods)) if len(methods) > 1 else [0]
                y_map = {label: i for i, label in enumerate(selected_labels)}
                for i, m in enumerate(methods):
                    mean_col, std_col = f"{m}_Mean", f"{m}_Std"
                    if mean_col in sub_df.columns:
                        fig.add_trace(
                            go.Scatter(x=sub_df[mean_col],
                                       y=[
                                           y_map.get(l) + offsets[i]
                                           for l in sub_df["display_label"]
                                       ],
                                       error_x=dict(type='data',
                                                    array=sub_df[std_col]),
                                       mode='markers',
                                       name=m,
                                       marker=dict(size=8)))
                fig.update_layout(
                    xaxis_range=[-1, 1],
                    yaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(selected_labels))),
                        ticktext=[l.split('] ')[1] for l in selected_labels]),
                    height=max(400,
                               len(selected_labels) * 50))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No index result files were found.")

    with tab_pie:
        st.header("Compositional Group Pie Charts")
        pie_groups = data.get('pie_groups', {})
        if sample_data_ref is not None and pie_groups:
            selected_group = st.selectbox("Select a Group",
                                          list(pie_groups.keys()),
                                          key="pie_select")
            tq_df, indices = data["tq_df"], pie_groups[selected_group]
            names, prog_cols = [tq_df["SHORT Q"].iloc[i] for i in indices], [
                get_prog_col_name(i, sample_data_ref) for i in indices
            ]
            target_sum = tq_df.iloc[indices[0]]['SUM50']
            cols = st.columns(len(available_methods))
            for i, m in enumerate(available_methods):
                with cols[i]:
                    st.subheader(f"{m} Method")
                    if (samples :=
                            data.get(f"samples_{m.lower()}")) is not None:
                        valid_cols = [
                            c for c in prog_cols if c in samples.columns
                        ]
                        if valid_cols:
                            means = samples[valid_cols].mean().values
                            norm_means = (means / means.sum()) * \
                                target_sum if means.sum() > 0 else means
                            pie_data = pd.DataFrame({
                                "Question": names,
                                "Mean Value": norm_means
                            })
                            fig_pie = px.pie(
                                pie_data,
                                values="Mean Value",
                                names="Question",
                                title=f"Proportions (Sum={target_sum})")
                            st.plotly_chart(fig_pie)
        else:
            st.warning("Sample files or compositional groups are required.")


if __name__ == '__main__':
    run()
