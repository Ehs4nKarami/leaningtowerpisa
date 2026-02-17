import pandas as pd
import plotly.graph_objects as go
import numpy as np


def classify_member(row):
    dx = row["EndX"] - row["StartX"]
    dy = row["EndY"] - row["StartY"]
    dz = row["EndZ"] - row["StartZ"]

    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0:
        return "unknown"

    angle = np.degrees(np.arccos(abs(dz) / length))

    if angle < 10:
        return "leg"           # almost vertical
    elif angle > 80:
        return "horizontal"    # almost flat
    else:
        return "brace"         # diagonal


def visualize_tower(csv_path):
    """
    Reads CSV with tower members, classifies them (leg, brace, horizontal),
    and creates an interactive 3D visualization with color coding.
    """
    df = pd.read_csv(csv_path)

    # Add classification column
    df["MemberClass"] = df.apply(classify_member, axis=1)

    # Color mapping for auto-classified members
    color_map = {
        'leg': 'red',
        'brace': 'green',
        'horizontal': 'blue',
        'unknown': 'black'
    }

    fig = go.Figure()

    # Group by classification
    for cls in df["MemberClass"].unique():
        df_cls = df[df["MemberClass"] == cls]

        x_coords, y_coords, z_coords = [], [], []

        for _, row in df_cls.iterrows():
            x_coords.extend([row['StartX'], row['EndX'], None])
            y_coords.extend([row['StartY'], row['EndY'], None])
            z_coords.extend([row['StartZ'], row['EndZ'], None])

        color = color_map.get(cls, 'black')

        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines',
            name=cls,
            line=dict(color=color, width=5),
            hovertemplate=f'{cls}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>'
        ))

    # Axis ranges
    x_range = [df[['StartX', 'EndX']].min().min(), df[['StartX', 'EndX']].max().max()]
    y_range = [df[['StartY', 'EndY']].min().min(), df[['StartY', 'EndY']].max().max()]
    z_range = [df[['StartZ', 'EndZ']].min().min(), df[['StartZ', 'EndZ']].max().max()]

    fig.update_layout(
        title='3D Tower Visualization (Colored by Member Class)',
        scene=dict(
            xaxis=dict(title='X (m)', range=x_range),
            yaxis=dict(title='Y (m)', range=y_range),
            zaxis=dict(title='Z (m)', range=z_range),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))
        ),
        width=1200,
        height=900,
        showlegend=True,
    )

    fig.show()

