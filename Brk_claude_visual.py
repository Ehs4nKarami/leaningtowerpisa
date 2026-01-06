import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re
from itertools import combinations

# Try to import trimesh for interference detection
try:
    import trimesh
    from trimesh.collision import CollisionManager
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("⚠️ Install trimesh for interference detection: pip install trimesh manifold3d")

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
    """
    vertices = [p1, p2, p3, p4, p5, p6, p7, p8]
    indices = [
        0, 2, 1,  0, 3, 2,  # Bottom
        4, 5, 6,  4, 6, 7,  # Top
        0, 1, 5,  0, 5, 4,  # Front
        1, 2, 6,  1, 6, 5,  # Right
        2, 3, 7,  2, 7, 6,  # Back
        3, 0, 4,  3, 4, 7   # Left
    ]
    return vertices, indices

# ==========================================
# 2. INTERFERENCE DETECTION FUNCTIONS
# ==========================================

def vertices_to_trimesh(vertices, indices):
    """
    Convert vertices and indices to a trimesh object.
    """
    if not vertices or not TRIMESH_AVAILABLE:
        return None
    
    try:
        verts = np.array(vertices)
        faces = np.array(indices).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.fix_normals()
        return mesh
    except Exception as e:
        print(f"Error creating mesh: {e}")
        return None

def check_mesh_collision(mesh_a, mesh_b):
    """
    Check if two meshes collide using collision manager.
    """
    if mesh_a is None or mesh_b is None:
        return False
    
    try:
        manager = CollisionManager()
        manager.add_object('a', mesh_a)
        manager.add_object('b', mesh_b)
        return manager.in_collision_internal()
    except:
        return False

def compute_intersection_mesh(mesh_a, mesh_b):
    """
    Compute the intersection volume between two meshes.
    Returns the intersection mesh or None.
    """
    if mesh_a is None or mesh_b is None:
        return None
    
    # Try different boolean engines
    engines = ['manifold', 'blender']
    
    for engine in engines:
        try:
            intersection = trimesh.boolean.intersection([mesh_a, mesh_b], engine=engine)
            if intersection is not None and hasattr(intersection, 'volume'):
                if intersection.volume > 0:
                    return intersection
        except Exception:
            continue
    
    return None

def detect_interferences(mesh_data_list, volume_threshold=1e-9, check_same_class=True):
    """
    Detect interferences between all mesh pairs.
    
    Parameters:
    -----------
    mesh_data_list : list of dicts
        Each dict contains: {'mesh': trimesh, 'index': int, 'class': str, 'row': series}
    volume_threshold : float
        Minimum intersection volume to report (filters tiny overlaps at connections)
    check_same_class : bool
        If True, also check collisions between same member classes
    
    Returns:
    --------
    list of interference dicts
    """
    if not TRIMESH_AVAILABLE:
        print("❌ trimesh not available. Cannot detect interferences.")
        return []
    
    print("\n🔍 Detecting interferences...")
    
    # Build collision manager with all meshes
    manager = CollisionManager()
    valid_indices = []
    
    for data in mesh_data_list:
        mesh = data['mesh']
        idx = data['index']
        if mesh is not None:
            try:
                manager.add_object(str(idx), mesh)
                valid_indices.append(idx)
            except Exception as e:
                pass
    
    print(f"   Checking {len(valid_indices)} meshes for collisions...")
    
    # Get all colliding pairs (broad phase)
    is_colliding, collision_pairs = manager.in_collision_internal(return_names=True)
    
    if not is_colliding:
        print("   ✅ No collisions detected!")
        return []
    
    print(f"   Found {len(collision_pairs)} potential collision pairs")
    
    # Build lookup dict
    mesh_lookup = {data['index']: data for data in mesh_data_list}
    
    interferences = []
    
    for name_a, name_b in collision_pairs:
        idx_a, idx_b = int(name_a), int(name_b)
        data_a = mesh_lookup.get(idx_a)
        data_b = mesh_lookup.get(idx_b)
        
        if data_a is None or data_b is None:
            continue
        
        # Optional: Skip same-class collisions
        if not check_same_class and data_a['class'] == data_b['class']:
            continue
        
        mesh_a = data_a['mesh']
        mesh_b = data_b['mesh']
        
        # Compute intersection mesh
        intersection_mesh = compute_intersection_mesh(mesh_a, mesh_b)
        
        if intersection_mesh is not None:
            volume = intersection_mesh.volume
            
            # Filter small intersections (connection points)
            if volume > volume_threshold:
                interferences.append({
                    'mesh': intersection_mesh,
                    'volume': volume,
                    'member_a_index': idx_a,
                    'member_b_index': idx_b,
                    'class_a': data_a['class'],
                    'class_b': data_b['class'],
                })
                print(f"   ⚠️ Interference: Member {idx_a} ({data_a['class']}) ↔ Member {idx_b} ({data_b['class']}) | Volume: {volume:.6e} m³")
        else:
            # Collision detected but couldn't compute intersection geometry
            # Still report it but without mesh
            interferences.append({
                'mesh': None,
                'volume': None,
                'member_a_index': idx_a,
                'member_b_index': idx_b,
                'class_a': data_a['class'],
                'class_b': data_b['class'],
            })
            print(f"   ⚠️ Interference (no mesh): Member {idx_a} ({data_a['class']}) ↔ Member {idx_b} ({data_b['class']})")
    
    print(f"\n📊 Total significant interferences: {len(interferences)}")
    return interferences

def create_interference_trace(interferences, color='magenta', opacity=0.95):
    """
    Create a Plotly Mesh3d trace for interference volumes.
    """
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    vertex_offset = 0
    
    for interference in interferences:
        mesh = interference.get('mesh')
        if mesh is None:
            continue
        
        verts = mesh.vertices
        faces = mesh.faces
        
        for v in verts:
            all_x.append(v[0])
            all_y.append(v[1])
            all_z.append(v[2])
        
        for face in faces:
            all_i.append(face[0] + vertex_offset)
            all_j.append(face[1] + vertex_offset)
            all_k.append(face[2] + vertex_offset)
        
        vertex_offset += len(verts)
    
    if not all_x:
        return None
    
    trace = go.Mesh3d(
        x=all_x, y=all_y, z=all_z,
        i=all_i, j=all_j, k=all_k,
        color=color,
        name=f'Interferences ({len(interferences)})',
        opacity=opacity,
        flatshading=True,
        lighting=dict(ambient=0.8, diffuse=0.9),
        hoverinfo='name',
    )
    
    return trace

def create_interference_markers(interferences, mesh_data_list):
    """
    Create scatter markers at interference locations for cases
    where mesh boolean failed.
    """
    mesh_lookup = {data['index']: data for data in mesh_data_list}
    
    x_points, y_points, z_points = [], [], []
    hover_texts = []
    
    for interference in interferences:
        if interference.get('mesh') is not None:
            # Already have mesh, use its centroid
            mesh = interference['mesh']
            centroid = mesh.centroid
            x_points.append(centroid[0])
            y_points.append(centroid[1])
            z_points.append(centroid[2])
        else:
            # No mesh, estimate position from member midpoints
            idx_a = interference['member_a_index']
            idx_b = interference['member_b_index']
            
            data_a = mesh_lookup.get(idx_a)
            data_b = mesh_lookup.get(idx_b)
            
            if data_a and data_b and 'row' in data_a and 'row' in data_b:
                row_a = data_a['row']
                row_b = data_b['row']
                
                mid_a = np.array([
                    (row_a['StartX'] + row_a['EndX']) / 2,
                    (row_a['StartY'] + row_a['EndY']) / 2,
                    (row_a['StartZ'] + row_a['EndZ']) / 2,
                ])
                mid_b = np.array([
                    (row_b['StartX'] + row_b['EndX']) / 2,
                    (row_b['StartY'] + row_b['EndY']) / 2,
                    (row_b['StartZ'] + row_b['EndZ']) / 2,
                ])
                
                mid = (mid_a + mid_b) / 2
                x_points.append(mid[0])
                y_points.append(mid[1])
                z_points.append(mid[2])
        
        vol_str = f"{interference['volume']:.2e}" if interference.get('volume') else "N/A"
        hover_texts.append(
            f"Member {interference['member_a_index']} ({interference['class_a']}) ↔ "
            f"Member {interference['member_b_index']} ({interference['class_b']})<br>"
            f"Volume: {vol_str} m³"
        )
    
    if not x_points:
        return None
    
    trace = go.Scatter3d(
        x=x_points, y=y_points, z=z_points,
        mode='markers',
        marker=dict(
            size=10,
            color='yellow',
            symbol='diamond',
            line=dict(color='red', width=2)
        ),
        name='Interference Points',
        hovertext=hover_texts,
        hoverinfo='text',
    )
    
    return trace

# ==========================================
# 3. LEG GENERATOR
# ==========================================

def generate_leg_mesh(row):
    start = np.array([row['StartX'], row['StartY'], row['StartZ']])
    end =   np.array([row['EndX'],   row['EndY'],   row['EndZ']])
    width, thickness = parse_section_dims(row['Section'])

    def get_inward_signs(x, y):
        dir_x = -1.0 if x >= 0 else 1.0
        dir_y = -1.0 if y >= 0 else 1.0
        return dir_x, dir_y

    sx_dir, sy_dir = get_inward_signs(start[0], start[1])
    ex_dir, ey_dir = get_inward_signs(end[0], end[1])

    def make_box1_corners(point, sign_x, sign_y):
        c0 = point
        c1 = point + np.array([width * sign_x, 0, 0])
        c2 = point + np.array([width * sign_x, thickness * sign_y, 0])
        c3 = point + np.array([0,              thickness * sign_y, 0])
        return c0, c1, c2, c3

    s_b1_0, s_b1_1, s_b1_2, s_b1_3 = make_box1_corners(start, sx_dir, sy_dir)
    e_b1_0, e_b1_1, e_b1_2, e_b1_3 = make_box1_corners(end, ex_dir, ey_dir)
    
    verts1, idx1 = create_cuboid(s_b1_0, s_b1_1, s_b1_2, s_b1_3, e_b1_0, e_b1_1, e_b1_2, e_b1_3)

    def make_box2_corners(point, sign_x, sign_y):
        c0 = point
        c1 = point + np.array([thickness * sign_x, 0, 0])
        c2 = point + np.array([thickness * sign_x, width * sign_y, 0])
        c3 = point + np.array([0,                  width * sign_y, 0])
        return c0, c1, c2, c3

    s_b2_0, s_b2_1, s_b2_2, s_b2_3 = make_box2_corners(start, sx_dir, sy_dir)
    e_b2_0, e_b2_1, e_b2_2, e_b2_3 = make_box2_corners(end, ex_dir, ey_dir)

    verts2, idx2 = create_cuboid(s_b2_0, s_b2_1, s_b2_2, s_b2_3, e_b2_0, e_b2_1, e_b2_2, e_b2_3)

    offset = len(verts1)
    verts_combined = verts1 + verts2
    idx_combined = idx1 + [i + offset for i in idx2]

    return verts_combined, idx_combined

# ==========================================
# 4. GENERIC BOX GENERATOR
# ==========================================

def get_oriented_box(row, width_mult=1.0):
    start = np.array([row['StartX'], row['StartY'], row['StartZ']])
    end =   np.array([row['EndX'],   row['EndY'],   row['EndZ']])
    width, _ = parse_section_dims(row['Section'])
    
    thickness = width * 0.2 
    width = width * width_mult

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

def generate_horizontal_mesh(row):
    return get_oriented_box(row)

def generate_brace_mesh(row):
    return get_oriented_box(row)

# ==========================================
# 5. MAIN VISUALIZER WITH INTERFERENCE CHECK
# ==========================================

def visualize_tower(csv_path, check_interferences=True, volume_threshold=1e-9):
    """
    Visualize tower with optional interference detection.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file
    check_interferences : bool
        Enable/disable interference checking
    volume_threshold : float
        Minimum volume to report as interference (m³)
    """
    df = pd.read_csv(csv_path)
    df["MemberClass"] = df.apply(classify_member, axis=1)

    fig = go.Figure()

    color_map = {
        'leg': 'red',
        'brace': '#00FF00', 
        'horizontal': 'blue',
        'unknown': 'grey'
    }

    print("🏗️ Generating 3D Geometry...")

    # Store all mesh data for interference checking
    all_mesh_data = []
    member_index = 0

    for cls in df["MemberClass"].unique():
        df_cls = df[df["MemberClass"] == cls]
        
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        vertex_offset = 0

        for _, row in df_cls.iterrows():
            
            if cls == 'leg':
                verts, idxs = generate_leg_mesh(row)
            elif cls == 'horizontal':
                verts, idxs = generate_horizontal_mesh(row)
            elif cls == 'brace':
                verts, idxs = generate_brace_mesh(row)
            else:
                verts, idxs = get_oriented_box(row)

            if not verts: 
                member_index += 1
                continue

            # Store mesh data for interference checking
            if check_interferences and TRIMESH_AVAILABLE:
                mesh = vertices_to_trimesh(verts, idxs)
                all_mesh_data.append({
                    'mesh': mesh,
                    'index': member_index,
                    'class': cls,
                    'row': row,
                    'vertices': verts,
                    'indices': idxs,
                })

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
            member_index += 1

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

    # ==========================================
    # INTERFERENCE DETECTION
    # ==========================================
    interferences = []
    
    if check_interferences and TRIMESH_AVAILABLE:
        interferences = detect_interferences(
            all_mesh_data, 
            volume_threshold=volume_threshold,
            check_same_class=True
        )
        
        # Add interference visualization
        if interferences:
            # Add intersection mesh trace
            interference_trace = create_interference_trace(
                interferences, 
                color='magenta', 
                opacity=0.9
            )
            if interference_trace:
                fig.add_trace(interference_trace)
            
            # Add marker points
            marker_trace = create_interference_markers(interferences, all_mesh_data)
            if marker_trace:
                fig.add_trace(marker_trace)

    # ==========================================
    # LAYOUT
    # ==========================================
    
    title = '3D Tower Visualization'
    if interferences:
        title += f' | ⚠️ {len(interferences)} Interferences Detected'
    elif check_interferences:
        title += ' | ✅ No Interferences'

    fig.update_layout(
        title=title,
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
    
    # Return interference report
    return {
        'total_members': member_index,
        'interferences': interferences,
        'interference_count': len(interferences)
    }

# ==========================================
# 6. INTERFERENCE REPORT GENERATOR
# ==========================================

def generate_interference_report(result, output_file=None):
    """
    Generate a detailed interference report.
    """
    print("\n" + "="*60)
    print("📋 INTERFERENCE REPORT")
    print("="*60)
    print(f"Total Members Analyzed: {result['total_members']}")
    print(f"Total Interferences: {result['interference_count']}")
    print("="*60)
    
    if result['interferences']:
        report_lines = []
        report_lines.append("Member A,Class A,Member B,Class B,Volume (m³)")
        
        for i, interference in enumerate(result['interferences'], 1):
            vol = interference.get('volume')
            vol_str = f"{vol:.6e}" if vol else "N/A"
            
            print(f"\n🔴 Interference #{i}")
            print(f"   Member A: {interference['member_a_index']} ({interference['class_a']})")
            print(f"   Member B: {interference['member_b_index']} ({interference['class_b']})")
            print(f"   Volume:   {vol_str} m³")
            
            report_lines.append(
                f"{interference['member_a_index']},{interference['class_a']},"
                f"{interference['member_b_index']},{interference['class_b']},{vol_str}"
            )
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(report_lines))
            print(f"\n💾 Report saved to: {output_file}")
    else:
        print("\n✅ No interferences detected!")
    
    print("="*60)

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    # Run visualization with interference check
    result = visualize_tower(
        "data.csv",
        check_interferences=True,
        volume_threshold=1e-9  # Adjust to filter small overlaps
    )
    
    # Generate report
    generate_interference_report(result, output_file="interference_report.csv")