from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.planwise_ml import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    health_label,
    load_bundle,
    score_tasks,
    train_all,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "planwise_ml_dataset.csv"
MODEL_PATH = ROOT / "artifacts" / "planwise_models.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

NAVY = "#102A43"
TEAL = "#007F7B"
CYAN = "#28B8B4"
AMBER = "#E8A020"
CORAL = "#D95852"
MIST = "#F4F7FA"
SLATE = "#60758A"
RISK_COLORS = {
    "None": "#AAB7C4",
    "Low": "#29A36A",
    "Medium": AMBER,
    "High": "#E87932",
    "Critical": CORAL,
}

st.set_page_config(page_title="PlanWise · Predict early, act wisely", page_icon="◈", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=Manrope:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --navy:#102A43; --teal:#007F7B; --mist:#F4F7FA; --line:#DCE6EE; }
        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color:var(--navy); }
        h1, h2, h3 { font-family:'Space Grotesk', sans-serif !important; letter-spacing:-0.035em; }
        [data-testid="stAppViewContainer"] { background:
          radial-gradient(circle at 82% -8%, rgba(40,184,180,.12), transparent 30rem), var(--mist); }
        [data-testid="stSidebar"] { background:#102A43; border-right:1px solid #21415e; }
        [data-testid="stSidebar"] * { color:#edf7f7; }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color:#102A43 !important; }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background:#F8FBFD; }
        [data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stMultiSelect label { font-weight:600; }
        .block-container { padding-top:2rem; padding-bottom:4rem; max-width:1540px; }
        .brand { display:flex; align-items:center; gap:.7rem; padding:.4rem 0 1.5rem; }
        .brand-mark { width:34px; height:34px; border-radius:11px; background:#28B8B4; color:#102A43;
          display:grid; place-items:center; font:700 18px 'Space Grotesk'; transform:rotate(-8deg); }
        .brand-name { font:700 24px 'Space Grotesk'; letter-spacing:-.04em; color:white; }
        .eyebrow { font:600 11px 'IBM Plex Mono'; letter-spacing:.13em; text-transform:uppercase; color:#007F7B; margin-bottom:.3rem; }
        .page-title { font:700 clamp(2.1rem, 4vw, 4rem)/.98 'Space Grotesk'; letter-spacing:-.06em; color:#102A43; margin:.1rem 0 .55rem; }
        .page-copy { color:#60758A; font-size:1rem; max-width:780px; margin-bottom:1.4rem; }
        .card { background:rgba(255,255,255,.92); border:1px solid #DCE6EE; border-radius:18px; padding:1.15rem 1.2rem;
          box-shadow:0 10px 30px rgba(16,42,67,.055); min-height:126px; }
        .card-label { font:600 10px 'IBM Plex Mono'; letter-spacing:.1em; text-transform:uppercase; color:#60758A; }
        .card-value { font:700 2rem 'Space Grotesk'; letter-spacing:-.04em; color:#102A43; margin:.35rem 0 .15rem; }
        .card-note { color:#60758A; font-size:.79rem; }
        .signal { background:#102A43; border-radius:20px; padding:1.2rem 1.4rem; color:white; overflow:hidden; position:relative; }
        .signal:after { content:''; position:absolute; width:180px; height:180px; right:-80px; top:-110px;
          border:28px solid rgba(40,184,180,.18); border-radius:50%; }
        .signal-label { font:600 10px 'IBM Plex Mono'; letter-spacing:.12em; text-transform:uppercase; color:#8ddbd7; }
        .signal-score { font:700 3.2rem 'Space Grotesk'; letter-spacing:-.06em; line-height:1; margin:.5rem 0; }
        .signal-track { height:8px; border-radius:8px; background:#294863; overflow:hidden; margin:.8rem 0 .5rem; }
        .signal-fill { height:100%; border-radius:8px; background:linear-gradient(90deg,#D95852,#E8A020,#28B8B4); }
        .pill { display:inline-block; padding:.28rem .55rem; border-radius:999px; background:#E7F6F5; color:#006E6A;
          font:600 11px 'IBM Plex Mono'; margin:.12rem .18rem .12rem 0; }
        .action-row { border-left:4px solid #E8A020; background:white; padding:.8rem 1rem; border-radius:0 12px 12px 0;
          margin:.55rem 0; box-shadow:0 4px 16px rgba(16,42,67,.04); }
        .action-title { font-weight:700; color:#102A43; }
        .action-copy { color:#60758A; font-size:.84rem; margin-top:.2rem; }
        .micro { font:500 11px 'IBM Plex Mono'; color:#60758A; }
        [data-testid="stMetric"] { background:white; border:1px solid #DCE6EE; padding:1rem; border-radius:16px; }
        [data-testid="stMetricValue"] { font-family:'Space Grotesk'; letter-spacing:-.04em; }
        .stButton button { border-radius:11px; min-height:44px; font-weight:700; border:1px solid #007F7B; }
        .stDownloadButton button { border-radius:11px; min-height:44px; font-weight:700; }
        div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input { border-radius:10px; min-height:44px; }
        [data-testid="stDataFrame"] { border:1px solid #DCE6EE; border-radius:14px; overflow:hidden; }
        a:focus, button:focus, input:focus, [tabindex]:focus { outline:3px solid rgba(40,184,180,.55) !important; outline-offset:2px; }
        @media (max-width: 800px) { .block-container { padding:1rem; } .page-title { font-size:2.5rem; } }
        @media (prefers-reduced-motion: reduce) { *, *:before, *:after { scroll-behavior:auto !important; transition:none !important; animation:none !important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return load_bundle(path)


@st.cache_data(show_spinner="Scoring the project portfolio…")
def cached_score(data: bytes, _bundle) -> pd.DataFrame:
    return score_tasks(pd.read_csv(io.BytesIO(data)), _bundle)


def page_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{kicker}</div><div class="page-title">{title}</div><div class="page-copy">{copy}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="card-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def base_figure(fig: go.Figure, height: int = 350) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=16, r=16, t=48, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.72)",
        font=dict(family="Manrope", color=NAVY),
        title_font=dict(family="Space Grotesk", size=18),
        legend_title_text="",
        hoverlabel=dict(bgcolor=NAVY, font_color="white"),
    )
    fig.update_xaxes(gridcolor="#E8EEF3", zeroline=False)
    fig.update_yaxes(gridcolor="#E8EEF3", zeroline=False)
    return fig


def aggregate_project_health(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("project_id", as_index=False).agg(
        project_health_score=("project_health_score", "mean"),
        delay_probability=("predicted_delay_probability", "mean"),
        task_count=("project_id", "size"),
        domain=("project_domain", "first"),
    )
    grouped["health"] = grouped["project_health_score"].map(health_label)
    return grouped


def render_overview(frame: pd.DataFrame) -> None:
    page_header(
        "Portfolio signal",
        "See around the corner.",
        "A live decision view of predicted delivery pressure, project risk, and where intervention has the highest value.",
    )
    projects = aggregate_project_health(frame)
    avg_health = float(projects["project_health_score"].mean())
    high_risk = int(frame["predicted_risk_level"].isin(["High", "Critical"]).sum())
    at_risk = int((frame["predicted_delay_probability"] >= st.session_state.bundle["threshold"]).sum())
    overloaded = int((frame["assignee_utilization"] >= 0.85).sum())

    left, right = st.columns([1.08, 2.2])
    with left:
        st.markdown(
            f"""<div class="signal"><div class="signal-label">Portfolio health signal</div>
            <div class="signal-score">{avg_health:.0f}<span style="font-size:1rem;color:#91a9ba"> / 100</span></div>
            <div style="font-weight:700">{health_label(avg_health)}</div>
            <div class="signal-track"><div class="signal-fill" style="width:{avg_health}%"></div></div>
            <div style="font-size:.78rem;color:#b8cbd8">Calculated from delay probability, risk, utilization, and dependencies.</div></div>""",
            unsafe_allow_html=True,
        )
    with right:
        cols = st.columns(3)
        with cols[0]:
            metric_card("Predicted delay", f"{at_risk:,}", f"{at_risk / max(len(frame),1):.1%} of visible tasks")
        with cols[1]:
            metric_card("High risk", f"{high_risk:,}", "High or critical classification")
        with cols[2]:
            metric_card("Capacity alerts", f"{overloaded:,}", "Tasks assigned above 85% utilization")

    st.markdown("### Portfolio map")
    c1, c2 = st.columns([1.35, 1])
    with c1:
        plot = px.scatter(
            projects,
            x="delay_probability",
            y="project_health_score",
            size="task_count",
            color="health",
            hover_name="domain",
            hover_data={"project_id": True, "task_count": True, "delay_probability": ":.1%", "project_health_score": ":.0f"},
            color_discrete_map={"Healthy": TEAL, "Watch": AMBER, "Critical": CORAL},
            category_orders={"health": ["Critical", "Watch", "Healthy"]},
            title="Projects by predicted pressure",
        )
        plot.update_xaxes(tickformat=".0%", title="Average delay probability")
        plot.update_yaxes(range=[0, 102], title="Health score")
        st.plotly_chart(base_figure(plot, 420), use_container_width=True)
    with c2:
        domain = frame.groupby("project_domain", as_index=False).agg(
            delay_probability=("predicted_delay_probability", "mean"), tasks=("project_domain", "size")
        ).sort_values("delay_probability")
        plot = px.bar(
            domain,
            x="delay_probability",
            y="project_domain",
            orientation="h",
            color="delay_probability",
            color_continuous_scale=[[0, "#BFE8E5"], [0.55, AMBER], [1, CORAL]],
            title="Delay pressure by domain",
            hover_data={"tasks": True, "delay_probability": ":.1%"},
        )
        plot.update_xaxes(tickformat=".0%", title="Predicted delay probability")
        plot.update_yaxes(title="")
        plot.update_layout(coloraxis_showscale=False)
        st.plotly_chart(base_figure(plot, 420), use_container_width=True)

    st.markdown("### Intervention queue")
    queue = frame.sort_values(
        ["project_health_score", "predicted_delay_probability"], ascending=[True, False]
    ).head(15)
    shown = [
        column
        for column in [
            "task_id",
            "project_id",
            "task_type",
            "assignee_role",
            "predicted_delay_probability",
            "predicted_risk_level",
            "project_health_score",
            "recommended_action",
        ]
        if column in queue
    ]
    st.dataframe(
        queue[shown],
        use_container_width=True,
        hide_index=True,
        column_config={
            "predicted_delay_probability": st.column_config.ProgressColumn("Delay probability", min_value=0, max_value=1, format="%.0%%"),
            "project_health_score": st.column_config.ProgressColumn("Health", min_value=0, max_value=100, format="%.0f"),
        },
    )


def render_workload(frame: pd.DataFrame) -> None:
    page_header(
        "Capacity intelligence",
        "Move work before work stalls.",
        "Find overloaded resources, compare roles, and turn model signals into practical allocation decisions.",
    )
    by_assignee = frame.groupby("assignee_id", as_index=False).agg(
        utilization=("assignee_utilization", "mean"),
        availability=("assignee_availability", "mean"),
        recent_hours=("assignee_recent_hours", "mean"),
        delay_probability=("predicted_delay_probability", "mean"),
        task_count=("assignee_id", "size"),
        role=("assignee_role", "first"),
    )
    by_assignee["capacity_state"] = np.select(
        [by_assignee["utilization"] >= 0.85, by_assignee["utilization"] <= 0.45],
        ["Overloaded", "Available"],
        default="Balanced",
    )
    cols = st.columns(3)
    with cols[0]:
        metric_card("Overloaded people", f"{(by_assignee['capacity_state'] == 'Overloaded').sum():,}", "Average utilization ≥ 85%")
    with cols[1]:
        metric_card("Available people", f"{(by_assignee['capacity_state'] == 'Available').sum():,}", "Average utilization ≤ 45%")
    with cols[2]:
        metric_card("Average utilization", f"{by_assignee['utilization'].mean():.0%}", "Across visible assignees")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        plot = px.scatter(
            by_assignee,
            x="utilization",
            y="delay_probability",
            size="task_count",
            color="capacity_state",
            hover_name="role",
            hover_data={"assignee_id": True, "recent_hours": ":.1f", "availability": ":.0%"},
            color_discrete_map={"Overloaded": CORAL, "Balanced": TEAL, "Available": "#6D8FB3"},
            title="Capacity versus delivery pressure",
        )
        plot.add_vline(x=0.85, line_dash="dash", line_color=CORAL, annotation_text="85% threshold")
        plot.update_xaxes(tickformat=".0%", title="Utilization")
        plot.update_yaxes(tickformat=".0%", title="Predicted delay probability")
        st.plotly_chart(base_figure(plot, 430), use_container_width=True)
    with c2:
        role = frame.groupby("assignee_role", as_index=False).agg(
            utilization=("assignee_utilization", "mean"),
            delay_probability=("predicted_delay_probability", "mean"),
            tasks=("assignee_role", "size"),
        ).sort_values("utilization")
        plot = px.bar(
            role,
            x="utilization",
            y="assignee_role",
            orientation="h",
            color="delay_probability",
            color_continuous_scale=[[0, "#BFE8E5"], [1, CORAL]],
            title="Role-level capacity",
            hover_data={"tasks": True, "delay_probability": ":.1%"},
        )
        plot.update_xaxes(tickformat=".0%", title="Average utilization")
        plot.update_yaxes(title="")
        plot.update_layout(coloraxis_showscale=False)
        st.plotly_chart(base_figure(plot, 430), use_container_width=True)

    st.markdown("### Recommended moves")
    urgent = frame[
        (frame["predicted_delay_probability"] >= 0.55)
        | (frame["assignee_utilization"] >= 0.85)
        | frame["predicted_risk_level"].isin(["High", "Critical"])
    ].sort_values(["project_health_score", "predicted_delay_probability"], ascending=[True, False]).head(8)
    if urgent.empty:
        st.success("No urgent workload moves in the current filter. Continue monitoring.")
    else:
        for _, row in urgent.iterrows():
            identifier = f"Task {int(row['task_id'])}" if "task_id" in row else row["task_type"]
            st.markdown(
                f'<div class="action-row"><div class="action-title">{identifier} · {row["task_type"]}</div>'
                f'<div class="action-copy">{row["recommended_action"]}</div>'
                f'<div class="micro">Delay {row["predicted_delay_probability"]:.0%} · Utilization {row["assignee_utilization"]:.0%} · Health {row["project_health_score"]:.0f}</div></div>',
                unsafe_allow_html=True,
            )


def number_input_for(column: str, profile: dict, key: str) -> float:
    values = profile["numeric"][column]
    integer_fields = {"project_budget", "project_start_days_ago", "estimated_duration_days", "num_dependencies", "dependency_depth", "num_risks", "max_risk_severity_ord"}
    is_integer = column in integer_fields
    step = 1 if is_integer else max(round((values["q75"] - values["q25"]) / 20, 2), 0.01)
    value = int(round(values["default"])) if is_integer else float(values["default"])
    return st.number_input(
        column.replace("_", " ").title(),
        min_value=int(np.floor(values["min"])) if is_integer else float(values["min"]),
        max_value=int(np.ceil(values["max"])) if is_integer else float(values["max"]),
        value=value,
        step=step,
        key=key,
    )


def render_prediction_lab(bundle) -> None:
    page_header(
        "Scenario lab",
        "Test a task before committing.",
        "Change scope, staffing, progress, or dependency assumptions and score the scenario with every trained model.",
    )
    profile = bundle["profile"]
    with st.form("prediction_form"):
        st.markdown("#### Project and task")
        columns = st.columns(3)
        values: dict[str, object] = {}
        for index, column in enumerate(CATEGORICAL_FEATURES):
            cfg = profile["categorical"][column]
            with columns[index % 3]:
                values[column] = st.selectbox(column.replace("_", " ").title(), cfg["options"], index=cfg["options"].index(cfg["default"]))
        st.markdown("#### Delivery conditions")
        columns = st.columns(4)
        for index, column in enumerate(NUMERIC_FEATURES):
            with columns[index % 4]:
                values[column] = number_input_for(column, profile, f"predict_{column}")
        submitted = st.form_submit_button("Score this scenario", use_container_width=True)

    if submitted:
        candidate = pd.DataFrame([values], columns=FEATURES)
        result = score_tasks(candidate, bundle).iloc[0]
        delay = float(result["predicted_delay_probability"])
        health = float(result["project_health_score"])
        st.markdown("### Decision signal")
        cols = st.columns(4)
        with cols[0]: metric_card("Delay probability", f"{delay:.0%}", f"Model threshold {bundle['threshold']:.0%}")
        with cols[1]: metric_card("Risk level", str(result["predicted_risk_level"]), f"Confidence {result['predicted_risk_confidence']:.0%}")
        with cols[2]: metric_card("Expected delay", f"{result['predicted_delay_days_if_delayed']:.1f} days", "Conditional on a delay")
        with cols[3]: metric_card("Health score", f"{health:.0f} / 100", health_label(health))
        st.info(f"Recommended action: {result['recommended_action']}")

        importance = pd.DataFrame(bundle["feature_importance"]["delay_classifier"]).head(10).sort_values("importance")
        plot = px.bar(importance, x="importance", y="feature", orientation="h", title="What the delay model weighs most")
        plot.update_traces(marker_color=TEAL)
        plot.update_xaxes(tickformat=".0%", title="Share of model importance")
        plot.update_yaxes(title="")
        st.plotly_chart(base_figure(plot, 360), use_container_width=True)
        st.caption("Feature importance explains the model globally; it is not a causal statement about this individual task.")


def render_model_performance(bundle, metrics: dict) -> None:
    page_header(
        "Model audit",
        "Trust is a measurable feature.",
        "Review holdout performance, tuning settings, confusion patterns, and the variables each Random Forest relies on.",
    )
    delay = metrics["delay_classifier"]
    risk = metrics["risk_classifier"]
    delay_days = metrics["delay_days_regressor"]
    overrun = metrics["effort_overrun_regressor"]
    cols = st.columns(4)
    with cols[0]: metric_card("Delay ROC–AUC", f"{delay['roc_auc']:.3f}", f"Recall {delay['recall']:.1%}")
    with cols[1]: metric_card("Risk macro F1", f"{risk['macro_f1']:.3f}", f"Balanced accuracy {risk['balanced_accuracy']:.1%}")
    with cols[2]: metric_card("Delay-days MAE", f"{delay_days['mae']:.2f}", f"R² {delay_days['r2']:.3f}")
    with cols[3]: metric_card("Overrun MAE", f"{overrun['mae']:.3f}", f"R² {overrun['r2']:.3f}")

    selected = st.selectbox(
        "Inspect model",
        ["delay_classifier", "risk_classifier", "delay_days_regressor", "effort_overrun_regressor"],
        format_func=lambda item: item.replace("_", " ").title(),
    )
    c1, c2 = st.columns([1.1, 1])
    with c1:
        importance = pd.DataFrame(bundle["feature_importance"][selected]).head(14).sort_values("importance")
        plot = px.bar(importance, x="importance", y="feature", orientation="h", title="Aggregated feature importance")
        plot.update_traces(marker_color=TEAL)
        plot.update_xaxes(tickformat=".0%", title="Importance")
        plot.update_yaxes(title="")
        st.plotly_chart(base_figure(plot, 450), use_container_width=True)
    with c2:
        if "confusion_matrix" in metrics[selected]:
            matrix = metrics[selected]["confusion_matrix"]
            labels = metrics[selected]["labels"]
            plot = px.imshow(
                matrix,
                x=[f"Predicted {label}" for label in labels],
                y=[f"Actual {label}" for label in labels],
                text_auto=True,
                color_continuous_scale=[[0, "#E7F3F3"], [1, TEAL]],
                title="Untouched test-set confusion matrix",
            )
            plot.update_layout(coloraxis_showscale=False)
            st.plotly_chart(base_figure(plot, 450), use_container_width=True)
        else:
            regression = metrics[selected]
            st.markdown("#### Holdout errors")
            st.metric("Mean absolute error", f"{regression['mae']:.3f}")
            st.metric("Root mean squared error", f"{regression['rmse']:.3f}")
            st.metric("R²", f"{regression['r2']:.3f}")

    with st.expander("Training protocol and best parameters"):
        metadata = bundle["metadata"]
        st.write(metadata["split_strategy"])
        st.write(metadata["leakage_controls"])
        st.write(metadata["tuning"])
        if selected == "risk_classifier" and metrics[selected]["macro_f1"] >= 0.999:
            st.warning(
                "The supplied dataset produces perfectly separable risk labels from its risk fields. "
                "Treat this as a property of the current synthetic data, not guaranteed real-world performance."
            )
        st.json(metrics[selected]["best_parameters"])


def render_data_workspace(raw: pd.DataFrame, frame: pd.DataFrame, bundle) -> None:
    page_header(
        "Data workspace",
        "Bring the next portfolio.",
        "Validate, score, export, or fully retrain on a compatible PlanWise dataset without changing application code.",
    )
    errors = validate_dataset(raw, require_targets=False)
    cols = st.columns(4)
    with cols[0]: metric_card("Rows", f"{len(raw):,}", "Tasks in current workspace")
    with cols[1]: metric_card("Columns", f"{raw.shape[1]:,}", f"{len(FEATURES)} model inputs required")
    with cols[2]: metric_card("Missing cells", f"{raw.isna().sum().sum():,}", "Handled by trained imputers")
    with cols[3]: metric_card("Duplicate rows", f"{raw.duplicated().sum():,}", "Removed during retraining")

    if errors:
        st.error(" ".join(errors))
    else:
        st.success("Schema validated. The dataset is ready for batch scoring.")

    scored_csv = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scored tasks",
        data=scored_csv,
        file_name="planwise_scored_tasks.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Retrain all four Random Forest models"):
        st.write("Retraining requires the four target columns and uses project-grouped train, validation, and untouched test sets.")
        iterations = st.slider("Tuning candidates per model", 3, 15, 6)
        can_train = not validate_dataset(raw, require_targets=True)
        if st.button("Retrain with current data", disabled=not can_train, use_container_width=True):
            retrain_dir = ROOT / "artifacts" / "retrained"
            progress = st.progress(0, text="Preparing grouped training splits…")
            iteration_log = st.empty()
            messages: list[str] = []

            def show_progress(event: dict) -> None:
                name = event["model"].replace("_", " ").title()
                if event["phase"] == "tuning":
                    message = (
                        f"{name}: iteration {event['iteration']}/{event['total_iterations']} · "
                        f"score {event['score']:.4f} · best {event['best_score']:.4f}"
                    )
                    messages.append(message)
                    progress.progress(
                        min(int(event["overall_fraction"] * 100), 99),
                        text=f"Model {event['model_index']}/{event['model_count']} · {message}",
                    )
                    iteration_log.code("\n".join(messages[-8:]), language=None)
                elif event["phase"] == "model_complete":
                    messages.append(f"✓ {event['message']}")
                    iteration_log.code("\n".join(messages[-8:]), language=None)

            result = train_all(
                raw,
                retrain_dir,
                iterations=iterations,
                progress_callback=show_progress,
            )
            progress.progress(100, text="Training complete · scoring the current dataset…")
            st.session_state.bundle = result["bundle"]
            st.session_state.metrics = result["metrics"]
            st.session_state.scored = score_tasks(raw, result["bundle"])
            st.success("Retraining complete. The dashboard now uses the new models for this session.")
            st.rerun()

    st.markdown("### Scored data preview")
    st.dataframe(frame.head(500), use_container_width=True, hide_index=True)


def main() -> None:
    inject_styles()
    if not MODEL_PATH.exists():
        st.error("Model artifacts are missing. Run `python train.py` before starting the dashboard.")
        st.stop()

    if "bundle" not in st.session_state:
        st.session_state.bundle = load_model(str(MODEL_PATH))
    if "metrics" not in st.session_state:
        st.session_state.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    bundle = st.session_state.bundle

    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">P</div><div class="brand-name">PlanWise</div></div>', unsafe_allow_html=True)
        view = st.radio(
            "Decision views",
            ["Portfolio overview", "Workload & actions", "Prediction lab", "Model performance", "Data workspace"],
        )
        st.divider()
        uploaded = st.file_uploader("Open another CSV", type=["csv"], help="Use the same feature schema for immediate batch scoring.")

    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        try:
            raw = pd.read_csv(io.BytesIO(uploaded_bytes))
            errors = validate_dataset(raw, require_targets=False)
            if errors:
                st.sidebar.error(" ".join(errors))
                raw = load_data(str(DATA_PATH))
                uploaded_bytes = raw.to_csv(index=False).encode("utf-8")
            else:
                st.sidebar.success(f"Loaded {len(raw):,} tasks")
        except Exception as exc:
            st.sidebar.error(f"Could not read the CSV: {exc}")
            raw = load_data(str(DATA_PATH))
            uploaded_bytes = raw.to_csv(index=False).encode("utf-8")
    else:
        raw = load_data(str(DATA_PATH))
        uploaded_bytes = raw.to_csv(index=False).encode("utf-8")

    scored = cached_score(uploaded_bytes, bundle)

    if view in {"Portfolio overview", "Workload & actions"}:
        with st.sidebar:
            st.markdown("#### Focus filters")
            domains = st.multiselect("Project domain", sorted(scored["project_domain"].astype(str).unique()))
            priorities = st.multiselect("Priority", sorted(scored["priority"].astype(str).unique()))
            statuses = st.multiselect("Task status", sorted(scored["status"].astype(str).unique()))
            risk_levels = st.multiselect("Predicted risk", [level for level in bundle["risk_order"] if level in scored["predicted_risk_level"].unique()])
        filtered = scored.copy()
        if domains: filtered = filtered[filtered["project_domain"].isin(domains)]
        if priorities: filtered = filtered[filtered["priority"].isin(priorities)]
        if statuses: filtered = filtered[filtered["status"].isin(statuses)]
        if risk_levels: filtered = filtered[filtered["predicted_risk_level"].isin(risk_levels)]
        st.sidebar.caption(f"{len(filtered):,} of {len(scored):,} tasks visible")
        if filtered.empty:
            st.warning("No tasks match these filters. Remove one or more filters to continue.")
            st.stop()
    else:
        filtered = scored

    if view == "Portfolio overview":
        render_overview(filtered)
    elif view == "Workload & actions":
        render_workload(filtered)
    elif view == "Prediction lab":
        render_prediction_lab(bundle)
    elif view == "Model performance":
        render_model_performance(bundle, st.session_state.metrics)
    else:
        render_data_workspace(raw, scored, bundle)

    st.caption("PlanWise supports project-manager decisions; it does not replace professional judgment. Predictions depend on data quality and should be monitored for drift.")


if __name__ == "__main__":
    main()
