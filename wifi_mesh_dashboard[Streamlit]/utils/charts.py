from __future__ import annotations

import math
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from utils.data_loader import DISTANCE_BANDS, DISTANCE_POINTS

COLORS = {
    "mesh": "#8b5cf6",
    "nomesh": "#38bdf8",
    "gain": "#34d399",
    "warn": "#f59e0b",
    "bad": "#f87171",
    "text": "#e5e7eb",
    "muted": "rgba(226,232,240,0.62)",
    "grid": "rgba(255,255,255,0.08)",
    "paper": "rgba(0,0,0,0)",
    "plot": "#11172b",
    "surface": "#0f172a",
}
ROUTER_COLORS = ["#8b5cf6", "#38bdf8", "#34d399", "#f59e0b", "#f87171", "#fb7185"]

_LAYOUT = dict(
    paper_bgcolor=COLORS["paper"],
    plot_bgcolor=COLORS["plot"],
    font=dict(color=COLORS["text"], family="Inter, Segoe UI, sans-serif", size=12),
    margin=dict(l=48, r=24, t=18, b=72),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.20,
        xanchor="left",
        x=0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    ),
    hoverlabel=dict(bgcolor="#111827", font=dict(color="#f8fafc")),
)

ROUTER_LABELS = {
    "Sagemcom_SSID": "Sagemcom",
    "TMO-G4AR": "G4AR",
    "TMO-G4SE": "G4SE",
    "TMO-G5AR": "G5AR",
    "KVD21": "KVD21",
}

def _router_label(name: str) -> str:
    return ROUTER_LABELS.get(name, name)
def _apply(fig: go.Figure, title: str | None = None, height: int = 320) -> go.Figure:
    fig.update_layout(height=height, **_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"])
    return fig

def _palette(i: int) -> str:
    return ROUTER_COLORS[i % len(ROUTER_COLORS)]

def line_chart(data: dict, routers: list[str], metric_unit: str, threshold: float | None = None) -> go.Figure:
    fig = go.Figure()

    router_labels = {
        "Sagemcom_SSID": "Sagemcom",
        "TMO-G4AR": "G4AR",
        "TMO-G4SE": "G4SE",
        "TMO-G5AR": "G5AR",
        "KVD21": "KVD21",
    }

    def label_of(name: str) -> str:
        return router_labels.get(name, name)

    for i, router in enumerate(routers):
        if router not in data:
            continue

        color = ROUTER_COLORS[i % len(ROUTER_COLORS)]
        vals = data[router]
        x = DISTANCE_POINTS
        mesh = vals["mesh"]
        nomesh = vals["nomesh"]
        gain = [(m - n) if (m == m and n == n) else np.nan for m, n in zip(mesh, nomesh)]
        name = label_of(router)

        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=mesh + nomesh[::-1],
            fill="toself",
            fillcolor="rgba(167,139,250,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
            name=f"{name} gap",
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=mesh,
            name=f"{name} · with mesh",
            mode="lines+markers",
            line=dict(color=color, width=3, shape="spline", smoothing=0.8),
            marker=dict(size=6),
            hovertemplate=f"{name}<br>With mesh: %{{y:.1f}} {metric_unit}<br>%{{x}} ft<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=nomesh,
            name=f"{name} · without mesh",
            mode="lines+markers",
            line=dict(color=color, width=2, dash="dot", shape="spline", smoothing=0.8),
            marker=dict(size=5),
            opacity=0.65,
            hovertemplate=f"{name}<br>Without mesh: %{{y:.1f}} {metric_unit}<br>%{{x}} ft<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=[x[-1]],
            y=[mesh[-1]],
            mode="markers",
            marker=dict(size=10, color=color, line=dict(width=2, color="#ffffff")),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Bar(
            x=x,
            y=gain,
            name=f"{name} gain",
            yaxis="y2",
            marker_color="rgba(52,211,153,0.18)",
            opacity=0.55,
            showlegend=False,
            hovertemplate=f"{name}<br>Gain: %{{y:.1f}} {metric_unit}<br>%{{x}} ft<extra></extra>",
        ))

    if threshold is not None:
        fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=COLORS["warn"],
    )

    fig.update_layout(
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["plot"],
        font=dict(color=COLORS["text"], family="sans-serif", size=12),
        margin=dict(l=48, r=48, t=18, b=72),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        xaxis=dict(
            title="Distance from router (ft)",
            tickvals=DISTANCE_POINTS,
            ticktext=[f"{d} ft" for d in DISTANCE_POINTS],
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(
            title=metric_unit,
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            title=f"Gain ({metric_unit})",
            color=COLORS["gain"],
        ),
        hovermode="x unified",
        height=430,
        barmode="overlay",
    )

    return fig

def grouped_bar(summary_df, unit: str) -> go.Figure:
    plot_df = summary_df.copy()
    plot_df["router_label"] = plot_df["router"].map(_router_label)

    show_text = len(plot_df) <= 3

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="With mesh",
        x=plot_df["router_label"],
        y=plot_df["avg_mesh"],
        marker=dict(color=COLORS["mesh"]),
        text=[f"{v:.0f}" if pd.notna(v) else "—" for v in plot_df["avg_mesh"]] if show_text else None,
        textposition="outside" if show_text else None,
        hovertemplate="%{x}<br>With mesh: %{y:.1f} " + unit + "<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="Without mesh",
        x=plot_df["router_label"],
        y=plot_df["avg_nomesh"],
        marker=dict(color=COLORS["nomesh"]),
        text=[f"{v:.0f}" if pd.notna(v) else "—" for v in plot_df["avg_nomesh"]] if show_text else None,
        textposition="outside" if show_text else None,
        hovertemplate="%{x}<br>Without mesh: %{y:.1f} " + unit + "<extra></extra>",
    ))

    fig.update_layout(
        barmode="group",
        xaxis=dict(title="Router"),
        yaxis=dict(title=unit),
    )

    return _apply(fig, height=360)

def radar_chart(kpis: dict) -> go.Figure:
    categories = ["Average", "Peak", "Coverage", "Low drop", "Consistency"]

    def norm(values):
        arr = np.asarray(values, dtype=float)
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if np.isclose(mx, mn):
            return [50.0] * len(arr)
        return [float((v - mn) / (mx - mn) * 100.0) for v in arr]

    routers = list(kpis.keys())
    avg_vals = [kpis[r]["avg_mesh"] for r in routers]
    peak_vals = [kpis[r]["peak_mesh"] for r in routers]
    cov_vals = [kpis[r]["coverage_mesh"] for r in routers]
    drop_vals = [kpis[r]["drop_mesh_total"] if "drop_mesh_total" in kpis[r] else sum(kpis[r]["drops_mesh"]) for r in routers]
    cons_vals = [kpis[r]["consistency_mesh"] for r in routers]

    avg_n = norm(avg_vals)
    peak_n = norm(peak_vals)
    cov_n = norm(cov_vals)
    drop_n = [100.0 - v for v in norm(drop_vals)]
    cons_n = [100.0 - v for v in norm(cons_vals)]

    fig = go.Figure()
    for i, router in enumerate(routers):
        vals = [avg_n[i], peak_n[i], cov_n[i], drop_n[i], cons_n[i]]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            name=router,
            line=dict(color=_palette(i), width=2),
            fill="toself",
            opacity=0.35,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["surface"],
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=COLORS["grid"]),
            angularaxis=dict(gridcolor=COLORS["grid"]),
        ),
    )
    return _apply(fig, "Router radar comparison", height=480)

def coverage_compare_bar(summary_df, unit: str) -> go.Figure:
    plot_df = summary_df.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="With mesh",
        x=plot_df["router"],
        y=plot_df["coverage_mesh"],
        marker=dict(color=COLORS["mesh"]),
        hovertemplate="%{x}<br>With mesh: %{y}/8 bands<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Without mesh",
        x=plot_df["router"],
        y=plot_df["coverage_nomesh"],
        marker=dict(color=COLORS["nomesh"]),
        hovertemplate="%{x}<br>Without mesh: %{y}/8 bands<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        xaxis=dict(title="Router"),
        yaxis=dict(title="Usable bands", range=[0, 8]),
    )
    return _apply(fig, "Coverage comparison", height=420)

def drop_off_compare(summary_df, unit: str) -> go.Figure:
    plot_df = summary_df.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="With mesh",
        x=plot_df["router"],
        y=plot_df["drop_mesh_total"],
        marker=dict(color=COLORS["mesh"]),
        hovertemplate="%{x}<br>With mesh drop: %{y:.1f} " + unit + "<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Without mesh",
        x=plot_df["router"],
        y=plot_df["drop_nomesh_total"],
        marker=dict(color=COLORS["bad"]),
        hovertemplate="%{x}<br>Without mesh drop: %{y:.1f} " + unit + "<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        xaxis=dict(title="Router"),
        yaxis=dict(title=f"Total drop ({unit})"),
    )
    return _apply(fig, "Drop-off by router", height=420)

def heatmap_chart(data: dict, topology: str = "mesh") -> go.Figure:
    key = "mesh" if topology == "mesh" else "nomesh"
    routers = list(data.keys())
    z = [[data[r][key][i] for i in range(len(DISTANCE_POINTS))] for r in routers]
    text = [[f"{v:.0f}" if not np.isnan(v) else "—" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"{d} ft" for d in DISTANCE_POINTS],
        y=routers,
        text=text,
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#1e1040"],
            [0.3, "#4f46e5"],
            [0.6, "#7c3aed"],
            [0.8, "#a78bfa"],
            [1.0, "#ddd6fe"],
        ],
        showscale=True,
        colorbar=dict(title="Value"),
        hovertemplate="%{y}<br>%{x}<br>%{z:.1f}<extra></extra>",
    ))
    return _apply(fig, f"Distance heatmap — {'with mesh' if key == 'mesh' else 'without mesh'}", height=460)

def horizontal_gain_bar(summary_df, unit: str) -> go.Figure:
    plot_df = summary_df.sort_values("gain_abs", ascending=True).copy()
    colors = [COLORS["gain"] if v >= 0 else COLORS["bad"] for v in plot_df["gain_abs"]]

    fig = go.Figure(go.Bar(
        x=plot_df["gain_abs"],
        y=plot_df["router"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.1f}" for v in plot_df["gain_abs"]],
        textposition="outside",
        hovertemplate="%{y}<br>Gain: %{x:.1f} " + unit + "<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title=f"Gain ({unit})"),
        yaxis=dict(title="", automargin=True),
    )
    return _apply(fig, "Mesh improvement ranking", height=580)