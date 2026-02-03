import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def classify_member(row):
    dx = row["EndX"] - row["StartX"]
    dy = row["EndY"] - row["StartY"]
    dz = row["EndZ"] - row["StartZ"]

    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0: return "unknown"

    angle = np.degrees(np.arccos(abs(dz) / length))

    if angle < 15:
        return "leg"
    elif angle > 80:
        return "horizontal"
    else:
        return "brace"

def parse_section_dims(section_str):
    """
    Parses 'L80x7' -> Width=0.08m, Thickness=0.007m
    Defaults to 50mm width, 5mm thickness if fails.
    """
    if not isinstance(section_str, str):
        return 0.05, 0.005
    
    nums = re.findall(r"(\d+)", section_str)
    
    width = 0.05
    thick = 0.005
    
    if len(nums) >= 1:
        width = float(nums[0]) / 1000.0 
        thick = width * 0.1             
    
    if len(nums) >= 2:
        thick = float(nums[1]) / 1000.0 

    return width, thick

def create_cuboid(p1, p2, p3, p4, p5, p6, p7, p8):
    """
    Creates vertices and indices for a box given 8 corner points.
    Order: Bottom 4 (CCW), Top 4 (CCW)
    """
    vertices = [p1, p2, p3, p4, p5, p6, p7, p8]
    indices = [
        0, 2, 1,  0, 3, 2, # Bottom
        4, 5, 6,  4, 6, 7, # Top
        0, 1, 5,  0, 5, 4, # Front
        1, 2, 6,  1, 6, 5, # Right
        2, 3, 7,  2, 7, 6, # Back
        3, 0, 4,  3, 4, 7  # Left
    ]
    return vertices, indices

# ==========================================
# 2. LEG GENERATOR (Aligned to Grid, Inward L)
# ==========================================

def generate_leg_mesh(row):
    start = np.array([row['StartX'], row['StartY'], row['StartZ']])
    end =   np.array([row['EndX'],   row['EndY'],   row['EndZ']])
    width, thickness = parse_section_dims(row['Section'])

    # Determine direction to "Inside"
    # We assume tower center is roughly (0,0). 
    # If Leg X is Positive, "Inside" is Negative direction.
    # If Leg X is Negative, "Inside" is Positive direction.
    
    def get_inward_signs(x, y):
        dir_x = -1.0 if x >= 0 else 1.0
        dir_y = -1.0 if y >= 0 else 1.0
        return dir_x, dir_y

    sx_dir, sy_dir = get_inward_signs(start[0], start[1])
    ex_dir, ey_dir = get_inward_signs(end[0], end[1])

    # We need to calculate the 4 corners of the profile at the Start Z
    # and the 4 corners at the End Z.
    
    # Vectors for Width and Thickness aligned with Global Axes
    # Flange 1: Parallel to X axis
    # Flange 2: Parallel to Y axis
    
    # --- BOX 1 (Aligned along X axis) ---
    # It starts at the corner.
    # Dimensions: Length = Width, Width = Thickness (Local Y)
    
    # Calculate corners relative to the Node Point
    def make_box1_corners(point, sign_x, sign_y):
        # Flange extending along X, with thickness in Y
        # Corner is at 'point'
        c0 = point
        c1 = point + np.array([width * sign_x, 0, 0])
        c2 = point + np.array([width * sign_x, thickness * sign_y, 0])
        c3 = point + np.array([0,              thickness * sign_y, 0])
        return c0, c1, c2, c3

    s_b1_0, s_b1_1, s_b1_2, s_b1_3 = make_box1_corners(start, sx_dir, sy_dir)
    e_b1_0, e_b1_1, e_b1_2, e_b1_3 = make_box1_corners(end, ex_dir, ey_dir)
    
    verts1, idx1 = create_cuboid(s_b1_0, s_b1_1, s_b1_2, s_b1_3, e_b1_0, e_b1_1, e_b1_2, e_b1_3)

    # --- BOX 2 (Aligned along Y axis) ---
    # It starts at the corner.
    # Dimensions: Length = Thickness (Local X), Width = Width (Local Y)
    
    def make_box2_corners(point, sign_x, sign_y):
        # Flange extending along Y, with thickness in X
        c0 = point
        c1 = point + np.array([thickness * sign_x, 0, 0])
        c2 = point + np.array([thickness * sign_x, width * sign_y, 0])
        c3 = point + np.array([0,                  width * sign_y, 0])
        return c0, c1, c2, c3

    s_b2_0, s_b2_1, s_b2_2, s_b2_3 = make_box2_corners(start, sx_dir, sy_dir)
    e_b2_0, e_b2_1, e_b2_2, e_b2_3 = make_box2_corners(end, ex_dir, ey_dir)

    verts2, idx2 = create_cuboid(s_b2_0, s_b2_1, s_b2_2, s_b2_3, e_b2_0, e_b2_1, e_b2_2, e_b2_3)

    # Combine meshes
    offset = len(verts1)
    verts_combined = verts1 + verts2
    idx_combined = idx1 + [i + offset for i in idx2]

    return verts_combined, idx_combined

# ==========================================
# 3. GENERIC BOX GENERATOR (For Braces/Horizontals)
# ==========================================

def get_oriented_box(row, width_mult=1.0):
    start = np.array([row['StartX'], row['StartY'], row['StartZ']])
    end =   np.array([row['EndX'],   row['EndY'],   row['EndZ']])
    width, _ = parse_section_dims(row['Section'])
    
    # Adjust width/thickness visuals
    thickness = width * 0.2 
    width = width * width_mult

    # Orientation Logic (Face Oriented)
    vec_long = end - start
    length = np.linalg.norm(vec_long)
    if length == 0: return [], []
    vec_long /= length
    
    mid = (start + end) / 2.0
    vec_radial = np.array([mid[0], mid[1], 0]) 
    if np.linalg.norm(vec_radial) == 0: vec_radial = np.array([1,0,0])
    else: vec_radial /= np.linalg.norm(vec_radial)
    
    vec_width = np.cross(vec_long, vec_radial)
    if np.linalg.norm(vec_width) < 0.1: vec_width = np.cross(vec_long, np.array([0,1,0]))
    vec_width /= np.linalg.norm(vec_width)
    
    vec_thick = np.cross(vec_long, vec_width)
    vec_thick /= np.linalg.norm(vec_thick)
    
    # Construct Corners
    w2 = width / 2.0
    t2 = thickness / 2.0
    
    c1 = -vec_width * w2 - vec_thick * t2
    c2 =  vec_width * w2 - vec_thick * t2
    c3 =  vec_width * w2 + vec_thick * t2
    c4 = -vec_width * w2 + vec_thick * t2
    
    s1 = start + c1
    s2 = start + c2
    s3 = start + c3
    s4 = start + c4
    
    vec_full = end - start
    e1 = s1 + vec_full
    e2 = s2 + vec_full
    e3 = s3 + vec_full
    e4 = s4 + vec_full

    return create_cuboid(s1, s2, s3, s4, e1, e2, e3, e4)

# ==========================================
# 4. SPECIFIC GENERATORS
# ==========================================

def generate_horizontal_mesh(row):
    # Logic for Horizontals (can be customized later)
    return get_oriented_box(row)

def generate_brace_mesh(row):
    # Logic for Braces (can be customized later)
    # Example: Making braces slightly thinner visually if desired?
    return get_oriented_box(row)

# ==========================================
# 5. MAIN VISUALIZER
# ==========================================

def visualize_tower(csv_path):
    df = pd.read_csv(csv_path)
    df["MemberClass"] = df.apply(classify_member, axis=1)

    fig = go.Figure()

    color_map = {
        'leg': 'red',
        'brace': '#00FF00', 
        'horizontal': 'blue',
        'unknown': 'grey'
    }

    print("Generating 3D Geometry...")

    for cls in df["MemberClass"].unique():
        df_cls = df[df["MemberClass"] == cls]
        
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        vertex_offset = 0

        for _, row in df_cls.iterrows():
            
            # --- SEPARATE CALLS FOR EACH TYPE ---
            if cls == 'leg':
                verts, idxs = generate_leg_mesh(row)
            elif cls == 'horizontal':
                verts, idxs = generate_horizontal_mesh(row)
            elif cls == 'brace':
                verts, idxs = generate_brace_mesh(row)
            else:
                verts, idxs = get_oriented_box(row)
            # ------------------------------------

            if not verts: continue

            for v in verts:
                all_x.append(v[0])
                all_y.append(v[1])
                all_z.append(v[2])
            
            num_triangles = len(idxs) // 3
            for t in range(num_triangles):
                all_i.append(idxs[t*3 + 0] + vertex_offset)
                all_j.append(idxs[t*3 + 1] + vertex_offset)
                all_k.append(idxs[t*3 + 2] + vertex_offset)

            vertex_offset += len(verts)

        color = color_map.get(cls, 'grey')

        fig.add_trace(go.Mesh3d(
            x=all_x, y=all_y, z=all_z,
            i=all_i, j=all_j, k=all_k,
            color=color,
            name=cls,
            opacity=1.0,
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.1) 
        ))

    fig.update_layout(
        title='3D Tower Visualization',
        scene=dict(
            xaxis=dict(title='X (m)', showgrid=False, backgroundcolor='rgb(240,240,240)'),
            yaxis=dict(title='Y (m)', showgrid=False, backgroundcolor='rgb(240,240,240)'),
            zaxis=dict(title='Z (m)', showgrid=False, backgroundcolor='rgb(230,230,245)'),
            aspectmode='data', 
            camera=dict(eye=dict(x=0.5, y=-0.5, z=0.2), up=dict(x=0,y=0,z=1))
        ),
        width=1200, height=900,
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.show()

if __name__ == "__main__":
    visualize_tower("data.csv")