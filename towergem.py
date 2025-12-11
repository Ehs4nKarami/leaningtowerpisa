import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re

def classify_member(row):
    """Classifies a member based on its geometry."""
    dx = row["EndX"] - row["StartX"]
    dy = row["EndY"] - row["StartY"]
    dz = row["EndZ"] - row["StartZ"]

    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0: return "unknown"

    angle = np.degrees(np.arccos(abs(dz) / length))

    if angle < 15: return "leg"           
    elif angle > 80: return "horizontal"    
    else: return "brace"         

def get_section_dims(section_str):
    """
    Parses 'L200x20'. Returns (width, thickness) in meters.
    Scale: 1000 assumes input is in mm.
    """
    if not isinstance(section_str, str): return 0.1, 0.01 

    match = re.search(r"(\d+\.?\d*)[xX](\d+\.?\d*)", section_str)
    if match:
        d1 = float(match.group(1))
        d2 = float(match.group(2))
        scale = 1000.0 
        return d1 / scale, d2 / scale
    
    return 0.1, 0.01 

def get_basis_vectors(p1, p2):
    """Calculates orientation vectors relative to tower center."""
    v_axis = p2 - p1
    length = np.linalg.norm(v_axis)
    if length == 0: return None, None, None
    v_axis = v_axis / length 

    # Radial vector (Center -> Member)
    midpoint = (p1 + p2) / 2.0
    v_radial = np.array([midpoint[0], midpoint[1], 0.0]) 
    
    if np.linalg.norm(v_radial) < 0.01:
        v_radial = np.array([1.0, 0.0, 0.0]) 
    else:
        v_radial = v_radial / np.linalg.norm(v_radial)

    # Tangent vector (v_tan)
    v_tan = np.cross(v_axis, v_radial)
    if np.linalg.norm(v_tan) < 0.01: v_tan = np.cross(v_axis, np.array([0,1,0]))
    v_tan = v_tan / np.linalg.norm(v_tan)

    # Inward vector (Normal to face, pointing IN towards center)
    # v_radial points OUT, so v_in points IN.
    v_in = -v_radial

    return v_axis, v_in, v_tan

def generate_brace_mesh(p1, p2, width, thickness):
    """Generates a flat cuboid for braces."""
    v_axis, v_in, v_tan = get_basis_vectors(p1, p2)
    if v_axis is None: return None, None

    # Braces: Width along Tangent, Thickness along In/Out
    v_flat = v_tan
    v_norm = v_in 
    
    w = width / 2.0
    t = thickness / 2.0

    corners = [
        p1 + v_flat*w + v_norm*t, p1 - v_flat*w + v_norm*t, 
        p1 - v_flat*w - v_norm*t, p1 + v_flat*w - v_norm*t,
        p2 + v_flat*w + v_norm*t, p2 - v_flat*w + v_norm*t, 
        p2 - v_flat*w - v_norm*t, p2 + v_flat*w - v_norm*t
    ]
    return np.array(corners), get_cube_indices()

def generate_leg_L_shape(p1, p2, width, thickness):
    """
    Generates an L-Shape pointing INWARD.
    p1 and p2 represent the OUTER CORNER (Heel).
    """
    v_axis, v_in, v_tan = get_basis_vectors(p1, p2)
    if v_axis is None: return None, None

    # Calculate flanges 45 degrees from the Inward vector
    # This aligns them with the faces of a square tower
    d1 = (v_in + v_tan)
    d1 = d1 / np.linalg.norm(d1)
    
    d2 = (v_in - v_tan)
    d2 = d2 / np.linalg.norm(d2)

    all_verts = []
    
    # --- Flange 1 ---
    # Starts at Corner (p1), extends Width along d1.
    # Thickness extends along d2 (Inward).
    f1_corners = [
        p1,                     p1 + d1*width, 
        p1 + d1*width + d2*thickness, p1 + d2*thickness,
        p2,                     p2 + d1*width, 
        p2 + d1*width + d2*thickness, p2 + d2*thickness
    ]
    all_verts.extend(f1_corners)

    # --- Flange 2 ---
    # Starts at Corner (p1), extends Width along d2.
    # Thickness extends along d1 (Inward).
    f2_corners = [
        p1,                     p1 + d2*width, 
        p1 + d2*width + d1*thickness, p1 + d1*thickness,
        p2,                     p2 + d2*width, 
        p2 + d2*width + d1*thickness, p2 + d1*thickness
    ]
    all_verts.extend(f2_corners)

    # Combine indices (Shift second box indices by 8)
    all_indices = get_cube_indices() + [x + 8 for x in get_cube_indices()]

    return np.array(all_verts), all_indices

def get_cube_indices():
    return [
        0, 1, 2,  0, 2, 3, # Front
        4, 6, 5,  4, 7, 6, # Back
        0, 4, 5,  0, 5, 1, # Top
        2, 6, 7,  2, 7, 3, # Bottom
        0, 3, 7,  0, 7, 4, # Right
        1, 5, 6,  1, 6, 2  # Left
    ]

def visualize_tower(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    cols = ['StartX', 'StartY', 'StartZ', 'EndX', 'EndY', 'EndZ']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=cols, inplace=True)
    df["MemberClass"] = df.apply(classify_member, axis=1)

    fig = go.Figure()

    # Mesh Lists
    leg_x, leg_y, leg_z = [], [], []
    leg_i, leg_j, leg_k = [], [], []
    leg_v_count = 0

    brace_x, brace_y, brace_z = [], [], []
    brace_i, brace_j, brace_k = [], [], []
    brace_v_count = 0

    # Hover Data Lists
    hover_x, hover_y, hover_z, hover_text = [], [], [], []

    print("Generating geometry...")

    for _, row in df.iterrows():
        p1 = np.array([row['StartX'], row['StartY'], row['StartZ']])
        p2 = np.array([row['EndX'], row['EndY'], row['EndZ']])
        w, t = get_section_dims(row.get('Section', ''))
        m_class = row["MemberClass"]

        verts = None
        idxs = None

        if m_class == "leg":
            # Generate L-Shape (Outer Corner = Line)
            verts, idxs = generate_leg_L_shape(p1, p2, w, t)
            
            if verts is not None:
                leg_x.extend(verts[:, 0])
                leg_y.extend(verts[:, 1])
                leg_z.extend(verts[:, 2])
                for m in range(0, len(idxs), 3):
                    leg_i.append(idxs[m] + leg_v_count)
                    leg_j.append(idxs[m+1] + leg_v_count)
                    leg_k.append(idxs[m+2] + leg_v_count)
                leg_v_count += len(verts)

        else:
            # Generate Flat Brace
            verts, idxs = generate_brace_mesh(p1, p2, w, t)
            
            if verts is not None:
                brace_x.extend(verts[:, 0])
                brace_y.extend(verts[:, 1])
                brace_z.extend(verts[:, 2])
                for m in range(0, len(idxs), 3):
                    brace_i.append(idxs[m] + brace_v_count)
                    brace_j.append(idxs[m+1] + brace_v_count)
                    brace_k.append(idxs[m+2] + brace_v_count)
                brace_v_count += len(verts)

        # Hover Data
        hover_x.extend([p1[0], p2[0], None])
        hover_y.extend([p1[1], p2[1], None])
        hover_z.extend([p1[2], p2[2], None])
        size_txt = f"{w*1000:.0f}x{t*1000:.0f}mm"
        info = f"ID: {row.get('ID', '?')}<br>Type: {m_class}<br>Sec: {row.get('Section', 'N/A')}<br>Size: {size_txt}"
        hover_text.extend([info, info, info])

    # --- Trace 1: Legs (Red L-Shapes) ---
    if leg_x:
        fig.add_trace(go.Mesh3d(
            x=leg_x, y=leg_y, z=leg_z,
            i=leg_i, j=leg_j, k=leg_k,
            color='red', name='Legs',
            flatshading=True, opacity=1.0,
            lighting=dict(ambient=0.6, diffuse=0.8),
            hoverinfo='skip'
        ))

    # --- Trace 2: Braces (Green) ---
    if brace_x:
        fig.add_trace(go.Mesh3d(
            x=brace_x, y=brace_y, z=brace_z,
            i=brace_i, j=brace_j, k=brace_k,
            color='green', name='Braces',
            flatshading=True, opacity=1.0,
            lighting=dict(ambient=0.5, diffuse=0.8),
            hoverinfo='skip'
        ))

    # --- Trace 3: Invisible Hover Lines ---
    fig.add_trace(go.Scatter3d(
        x=hover_x, y=hover_y, z=hover_z,
        mode='lines', name='Info',
        line=dict(color='white', width=2), opacity=0.0,
        text=hover_text, hoverinfo='text'
    ))

    fig.update_layout(
        title='3D Tower Visualization (True Scale)',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data', # Critical for correct proportions
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
        )
    )

    fig.show()

if __name__ == "__main__":
    visualize_tower("data.csv")