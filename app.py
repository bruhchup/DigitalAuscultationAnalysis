"""
=============================================================================
AusculTek - Streamlit Web Application
=============================================================================

Web-based interface for respiratory sound classification.
Wraps the AudioClassifier inference pipeline with an interactive dashboard.

Usage:
  streamlit run app.py

Requires:
  - inference.py (AudioClassifier) in the same directory
  - models/best_model/ with model.pkl, scaler.pkl, preprocessing_config.json
  - pip install streamlit plotly librosa numpy scipy

Author: Hayden Banks
Date: February 2026
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import librosa
import tempfile
import os
import json
from pathlib import Path
from datetime import datetime

# Import classifier
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "audio_preprocessing"))
from inference import AudioClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = os.environ.get("MODEL_DIR", "models/best_model")
CLASS_COLORS = {
    "Normal": "#0891B2",
    "Crackle": "#F97316",
    "Wheeze": "#EF4444",
}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AusculTek",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #F0F9FF;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0C4A6E;
    }
    [data-testid="stSidebar"] * {
        color: #E0F2FE !important;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #FFFFFF !important;
    }

    /* File uploader labels inside sidebar - dark teal for contrast */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label *,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] div {
        color: #0C4A6E !important;
    }

    /* File uploader drop zone - white background, dark text */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: #0C4A6E !important;
    }

    /* Browse files button - gray background */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background-color: #E2E8F0 !important;
        color: #0C4A6E !important;
        border: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #CBD5E1 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    [data-testid="stMetric"] label {
        color: #64748B !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0C4A6E !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #0C4A6E !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* Divider */
    hr {
        border-color: #CBD5E1 !important;
    }

    /* Hide default streamlit footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_classifier():
    """Load the classifier once and cache it."""
    try:
        clf = AudioClassifier(MODEL_DIR)
        return clf, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def classify_audio(clf, audio_path, annotation_path=None):
    """Run classification and return results."""
    return clf.classify(audio_path, annotation_path)


def create_overall_gauge(label, confidence):
    """Create a gauge chart for overall classification."""
    color = CLASS_COLORS.get(label, "#64748B")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        title={"text": f"<b>{label}</b>", "font": {"size": 22, "color": "#0C4A6E"}},
        number={"suffix": "%", "font": {"size": 36, "color": "#0C4A6E"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "#F1F5F9",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "#FEE2E2"},
                {"range": [33, 66], "color": "#FEF3C7"},
                {"range": [66, 100], "color": "#D1FAE5"},
            ],
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Calibri, sans-serif", "color": "#0C4A6E"},
    )
    return fig


def create_class_distribution_pie(results):
    """Create a pie chart of segment classifications."""
    counts = {"Normal": 0, "Crackle": 0, "Wheeze": 0}
    for c in results["cycles"]:
        if c["label"] in counts:
            counts[c["label"]] += 1

    labels = [k for k, v in counts.items() if v > 0]
    values = [v for v in counts.values() if v > 0]
    colors = [CLASS_COLORS.get(l, "#64748B") for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="label+value",
        textfont=dict(size=13),
        hovertemplate="<b>%{label}</b><br>Segments: %{value}<br>%{percent}<extra></extra>",
    )])
    fig.update_layout(
        title=dict(text="Segment Classification", font=dict(size=16, color="#0C4A6E")),
        height=320,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
            font=dict(color="#0C4A6E", size=13),
        ),
        font={"family": "Calibri, sans-serif", "color": "#0C4A6E"},
    )
    return fig


def create_confidence_timeline(results):
    """Create a bar chart showing per-segment confidence and classification."""
    if not results["cycles"]:
        return None

    seg_labels = [f"{c['start']:.1f}-{c['end']:.1f}s" for c in results["cycles"]]
    labels = [c["label"] for c in results["cycles"]]
    confs = [c["confidence"] * 100 for c in results["cycles"]]
    colors = [CLASS_COLORS.get(l, "#64748B") for l in labels]

    fig = go.Figure(data=go.Bar(
        x=seg_labels,
        y=confs,
        marker_color=colors,
        hovertemplate=[
            f"<b>{l}</b><br>Time: {s}<br>Confidence: {c:.1f}%<extra></extra>"
            for l, s, c in zip(labels, seg_labels, confs)
        ],
    ))

    fig.update_layout(
        title=dict(text="Confidence by Segment", font=dict(size=16, color="#0C4A6E")),
        xaxis_title="Time Segment",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 105]),
        height=300,
        margin=dict(l=50, r=20, t=50, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font={"family": "Calibri, sans-serif", "color": "#0C4A6E"},
    )
    fig.update_xaxes(showgrid=False, linecolor="#0C4A6E", linewidth=1, tickangle=-45,
                     tickfont=dict(color="#0C4A6E"), title_font=dict(color="#0C4A6E"))
    fig.update_yaxes(gridcolor="#CBD5E1", gridwidth=0.5, linecolor="#0C4A6E", linewidth=1,
                     tickfont=dict(color="#0C4A6E"), title_font=dict(color="#0C4A6E"))

    return fig


def create_probability_heatmap(results):
    """Create a heatmap of class probabilities across segments."""
    if not results["cycles"]:
        return None

    classes = list(results["cycles"][0]["probabilities"].keys())
    n_segments = len(results["cycles"])

    z_data = []
    for cls in classes:
        row = [c["probabilities"][cls] * 100 for c in results["cycles"]]
        z_data.append(row)

    seg_labels = [f"{c['start']:.1f}-{c['end']:.1f}s" for c in results["cycles"]]

    colors = [[0, "#F0F9FF"], [0.25, "#BAE6FD"], [0.5, "#38BDF8"],
              [0.75, "#0891B2"], [1, "#0C4A6E"]]

    # Dynamic text color: white on dark cells, dark on light cells
    text_colors = []
    for row in z_data:
        text_row = []
        for v in row:
            text_row.append("#FFFFFF" if v > 40 else "#0C4A6E")
        text_colors.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=seg_labels,
        y=classes,
        colorscale=colors,
        hovertemplate="<b>%{y}</b><br>Segment: %{x}<br>Probability: %{z:.1f}%<extra></extra>",
        colorbar=dict(title=dict(text="Prob %", side="right")),
        showscale=True,
    ))

    # Add per-cell text annotations with contrast colors
    for i, cls in enumerate(classes):
        for j, seg in enumerate(seg_labels):
            fig.add_annotation(
                x=seg, y=cls,
                text=f"{z_data[i][j]:.0f}%",
                showarrow=False,
                font=dict(size=12, color=text_colors[i][j]),
            )

    fig.update_layout(
        title=dict(text="Class Probabilities Across Segments", font=dict(size=16, color="#0C4A6E")),
        height=250,
        margin=dict(l=80, r=20, t=50, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        xaxis_title="Time Segment",
        xaxis=dict(tickfont=dict(color="#0C4A6E"), title_font=dict(color="#0C4A6E")),
        yaxis=dict(tickfont=dict(color="#0C4A6E"), title_font=dict(color="#0C4A6E")),
        font={"family": "Calibri, sans-serif", "color": "#0C4A6E"},
    )
    return fig


def create_waveform_plot(audio, sr, results):
    """Create an annotated waveform visualization."""
    time_axis = np.linspace(0, len(audio) / sr, len(audio))

    fig = go.Figure()

    # Base waveform
    # Downsample for performance if long
    step = max(1, len(audio) // 5000)
    fig.add_trace(go.Scatter(
        x=time_axis[::step],
        y=audio[::step],
        mode="lines",
        line=dict(color="#94A3B8", width=0.8),
        name="Waveform",
        hoverinfo="skip",
    ))

    # Overlay colored regions for each classified segment
    if results and results["cycles"]:
        for c in results["cycles"]:
            color = CLASS_COLORS.get(c["label"], "#64748B")
            fig.add_vrect(
                x0=c["start"], x1=c["end"],
                fillcolor=color, opacity=0.15, line_width=0,
            )

    fig.update_layout(
        title=dict(text="Waveform with Classification Overlay", font=dict(size=16, color="#0C4A6E")),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=250,
        margin=dict(l=50, r=20, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        showlegend=False,
        font={"family": "Calibri, sans-serif", "color": "#0C4A6E"},
    )
    fig.update_xaxes(showgrid=False, linecolor="#0C4A6E", linewidth=1,
                     tickfont=dict(color="#0C4A6E"), title_font=dict(color="#0C4A6E"))
    fig.update_yaxes(showgrid=True, gridcolor="#CBD5E1", gridwidth=0.5, linecolor="#0C4A6E", linewidth=1,
                     tickfont=dict(color="#0C4A6E"), title_font=dict(color="#0C4A6E"))

    return fig


# ---------------------------------------------------------------------------
# Session state for history
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("# AusculTek")
    st.markdown("**Respiratory Sound Analysis**")
    st.markdown("---")

    st.markdown("### Upload Audio")
    uploaded_file = st.file_uploader(
        "Upload a .wav lung recording",
        type=["wav"],
        help="Upload a .wav file recorded with a digital stethoscope",
    )

    # Optional ICBHI annotations
    annotation_file = st.file_uploader(
        "ICBHI annotations (optional)",
        type=["txt"],
        help="Upload the matching .txt annotation file for ICBHI recordings",
    )

    st.markdown("---")

    st.markdown("### Session History")
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history)):
            label = entry["overall_label"]
            color = CLASS_COLORS.get(label, "#64748B")
            icon = "+" if label == "Normal" else "!"
            st.markdown(
                f"{icon} **{entry['filename']}**  \n"
                f"<span style='color:{color}'>{label}</span> · "
                f"{entry['confidence']:.0%} · {entry['time']}",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("*No analyses yet*")

    st.markdown("---")
    st.markdown(
        "<small style='color:#94A3B8'>CIS 485 · Spring 2026</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# Load classifier
clf, load_error = load_classifier()

if load_error:
    st.error(f"**Model failed to load:** {load_error}")
    st.info(
        f"Make sure `{MODEL_DIR}/` contains `model.pkl`, `scaler.pkl`, "
        f"and `preprocessing_config.json`."
    )
    st.stop()

# Header
col_title, col_status = st.columns([3, 1])
with col_title:
    greeting = "Good Morning" if datetime.now().hour < 12 else (
        "Good Afternoon" if datetime.now().hour < 17 else "Good Evening"
    )
    st.markdown(f"## {greeting}")
with col_status:
    st.markdown(
        "<div style='text-align:right; padding-top:12px'>"
        "<span style='background:#D1FAE5; color:#065F46; padding:4px 12px; "
        "border-radius:12px; font-size:13px'>Model Ready</span></div>",
        unsafe_allow_html=True,
    )

# ---- No file uploaded: show landing ----
if uploaded_file is None:
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "<div style='background:white; padding:24px; border-radius:12px; "
            "box-shadow:0 1px 3px rgba(0,0,0,0.08); text-align:center'>"
            "<h3 style='margin-bottom:8px'>Upload</h3>"
            "<p style='color:#64748B'>Upload a .wav lung recording from the sidebar</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            "<div style='background:white; padding:24px; border-radius:12px; "
            "box-shadow:0 1px 3px rgba(0,0,0,0.08); text-align:center'>"
            "<h3 style='margin-bottom:8px'>Analyze</h3>"
            "<p style='color:#64748B'>ML classifier detects crackles & wheezes</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            "<div style='background:white; padding:24px; border-radius:12px; "
            "box-shadow:0 1px 3px rgba(0,0,0,0.08); text-align:center'>"
            "<h3 style='margin-bottom:8px'>Review</h3>"
            "<p style='color:#64748B'>Interactive dashboard with per-segment results</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#94A3B8; font-size:14px'>"
        "Upload a recording to begin analysis</p>",
        unsafe_allow_html=True,
    )
    st.stop()

# ---- File uploaded: run classification ----
st.markdown("---")

# Save uploaded file to temp
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tmp.write(uploaded_file.getvalue())
    tmp_path = tmp.name

# Save annotation file if provided
annotation_path = None
if annotation_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_ann:
        tmp_ann.write(annotation_file.getvalue())
        annotation_path = tmp_ann.name

# Classify
with st.spinner("Analyzing respiratory sounds..."):
    results = classify_audio(clf, tmp_path, annotation_path)

    # Load audio for waveform display
    audio, sr = librosa.load(tmp_path, sr=8000)

# Clean up temp files
os.unlink(tmp_path)
if annotation_path:
    os.unlink(annotation_path)

# Check for errors
if results.get("error"):
    st.error(f"Classification error: {results['error']}")
    st.stop()

# Add to history
st.session_state.history.append({
    "filename": uploaded_file.name,
    "overall_label": results["overall_label"],
    "confidence": results["overall_confidence"],
    "time": datetime.now().strftime("%I:%M %p"),
    "results": results,
})

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

# Row 1: Summary metrics
st.markdown("### Analysis Results")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Overall Classification", results["overall_label"])
with col2:
    st.metric("Confidence", f"{results['overall_confidence']:.0%}")
with col3:
    st.metric("Segments Analyzed", results["total_cycles"])
with col4:
    st.metric("Duration", f"{results['duration_sec']:.1f}s")

st.markdown("")

# Row 2: Gauge + Pie + Abnormal count
col_gauge, col_pie, col_stats = st.columns([1, 1, 1])

with col_gauge:
    fig_gauge = create_overall_gauge(results["overall_label"], results["overall_confidence"])
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_pie:
    fig_pie = create_class_distribution_pie(results)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_stats:
    st.markdown("")
    st.markdown("")

    abnormal_pct = (
        results["abnormal_cycles"] / results["total_cycles"] * 100
        if results["total_cycles"] > 0 else 0
    )

    if results["overall_label"] == "Normal":
        st.markdown(
            f"<div style='background:#D1FAE5; padding:12px 16px; border-radius:8px; "
            f"color:#0C4A6E; border-left:4px solid #0891B2'>"
            f"<b>No abnormalities detected</b> across {results['total_cycles']} segments.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background:#FEF3C7; padding:12px 16px; border-radius:8px; "
            f"color:#0C4A6E; border-left:4px solid #F97316'>"
            f"<b>{results['abnormal_cycles']} of {results['total_cycles']}</b> segments "
            f"({abnormal_pct:.0f}%) show abnormalities.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.metric("Normal Segments", results["normal_cycles"])
    st.metric("Abnormal Segments", results["abnormal_cycles"])

# Row 3: Waveform
st.markdown("---")
fig_waveform = create_waveform_plot(audio, sr, results)
st.plotly_chart(fig_waveform, use_container_width=True)

# Row 4: Heatmap + Timeline
col_heat, col_time = st.columns(2)

with col_heat:
    fig_heatmap = create_probability_heatmap(results)
    if fig_heatmap:
        st.plotly_chart(fig_heatmap, use_container_width=True)

with col_time:
    fig_timeline = create_confidence_timeline(results)
    if fig_timeline:
        st.plotly_chart(fig_timeline, use_container_width=True)

# Row 5: Detailed segment table
st.markdown("---")
st.markdown("### Segment Details")

# Build table data
table_data = []
for i, c in enumerate(results["cycles"]):
    p = c["probabilities"]
    table_data.append({
        "Segment": i + 1,
        "Start (s)": f"{c['start']:.2f}",
        "End (s)": f"{c['end']:.2f}",
        "Classification": c["label"],
        "Confidence": f"{c['confidence']:.1%}",
        "Normal %": f"{p.get('Normal', 0):.1%}",
        "Crackle %": f"{p.get('Crackle', 0):.1%}",
        "Wheeze %": f"{p.get('Wheeze', 0):.1%}",
    })

st.dataframe(
    table_data,
    use_container_width=True,
    hide_index=True,
)

# Row 6: Raw JSON (collapsible)
with st.expander("Raw JSON Output"):
    st.json(results)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#94A3B8; font-size:12px'>"
    "AusculTek · Respiratory Sound Classification · CIS 485 Spring 2026 · "
    "Hayden Banks, Jared Tauler</p>",
    unsafe_allow_html=True,
)