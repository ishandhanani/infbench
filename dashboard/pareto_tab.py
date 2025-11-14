"""
Pareto Frontier Analysis Tab
"""

import pandas as pd
import streamlit as st

from srtslurm.visualizations import calculate_pareto_frontier, create_pareto_graph


def render(df: pd.DataFrame, selected_runs: list[str], run_legend_labels: dict, pareto_options: dict):
    """Render the Pareto frontier analysis tab.
    
    Args:
        df: DataFrame with benchmark data
        selected_runs: List of run IDs
        run_legend_labels: Dict mapping run_id to display label
        pareto_options: Dict with show_cutoff, cutoff_value, show_frontier
    """
    st.subheader("Pareto Frontier Analysis")
    
    # Y-axis metric toggle
    y_axis_metric = st.radio(
        "Y-axis metric",
        options=["Output TPS/GPU", "Total TPS/GPU"],
        index=0,
        horizontal=True,
        help="Choose between decode throughput per GPU or total throughput per GPU (input + output)",
    )
    
    if y_axis_metric == "Total TPS/GPU":
        st.markdown("""
        This graph shows the trade-off between **Total TPS/GPU** (input + output tokens/s per GPU) and
        **Output TPS/User** (throughput per user).
        """)
    else:
        st.markdown("""
        This graph shows the trade-off between **Output TPS/GPU** (decode tokens/s per GPU) and
        **Output TPS/User** (throughput per user).
        """)
    
    pareto_fig = create_pareto_graph(
        df,
        selected_runs,
        pareto_options["show_cutoff"],
        pareto_options["cutoff_value"],
        pareto_options["show_frontier"],
        y_axis_metric,
        run_legend_labels,
    )
    pareto_fig.update_xaxes(showgrid=True)
    pareto_fig.update_yaxes(showgrid=True)
    
    st.plotly_chart(pareto_fig, width="stretch", key="pareto_main")
    
    # Debug info for frontier
    if pareto_options["show_frontier"]:
        frontier_points = calculate_pareto_frontier(df, y_axis_metric)
        st.caption(f"🔍 Debug: Frontier has {len(frontier_points)} points across {len(df)} total data points")
        
        if len(frontier_points) > 0:
            with st.expander("View Frontier Points Details"):
                frontier_df = pd.DataFrame(frontier_points, columns=["Output TPS/User", "Output TPS/GPU"])
                st.dataframe(frontier_df, width="stretch")
    
    # Data export button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.download_button(
            label="📥 Download Data as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="benchmark_data.csv",
            mime="text/csv",
            width="stretch",
        )
    
    # Metric calculation documentation
    st.divider()
    st.markdown("### 📊 Metric Calculations")
    st.markdown("""
    **How each metric is calculated:**

    **Output TPS/GPU** (Throughput Efficiency):
    """)
    st.latex(
        r"\text{Output TPS/GPU} = \frac{\text{Total Output Throughput (tokens/s)}}{\text{Total Number of GPUs}}"
    )
    st.markdown("""
    *This measures how efficiently each GPU is being utilized for token generation.*

    **Output TPS/User** (Per-User Generation Rate):
    """)
    st.latex(r"\text{Output TPS/User} = \frac{1000}{\text{Mean TPOT (ms)}}")
    st.markdown("""
    *Where TPOT (Time Per Output Token) is the average time between consecutive output tokens.
    This represents the actual token generation rate experienced by each user, independent of concurrency.*
    """)

