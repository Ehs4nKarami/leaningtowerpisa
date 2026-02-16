import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re
import csv

# ==========================================================
# CONSTANTS
# ==========================================================

HOLE_DIAMETER = 0.012
HOLE_RADIUS = HOLE_DIAMETER / 2
HOLE_OFFSET_FACTOR = 1.5
HOLE_CYLINDER_LENGTH = 0.05

# ==========================================================
# CLASSIFICATION
# ==========================================================

def classify_member(row):
    dx = row["EndX"] - row["StartX"]
    dy = row["EndY"] - row["StartY"]
    dz = row["EndZ"] - row["StartZ"]

    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0:
        return "unknown"

    angle = np.degrees(np.arccos(abs(dz) / length))

    if angle < 15:
        return "leg"
    elif angle > 80:
        return "horizontal"
    else:
        return "brace"

# ==========================================================
# VECTOR UTILITIES
# ==========================================================

def vec(row):
    return np.array([
        row['EndX'] - row['StartX'],
        row['EndY'] - row['StartY'],
        row['EndZ'] - row['StartZ']
    ])

def unit(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def midpoint(row):
    return np.array([
        (row['StartX'] + row['EndX']) / 2,
        (row['StartY'] + row['EndY']) / 2,
        (row['StartZ'] + row['EndZ']) / 2
    ])

def calculate_length(row):
    """Calculate the length of a member"""
    dx = row['EndX'] - row['StartX']
    dy = row['EndY'] - row['StartY']
    dz = row['EndZ'] - row['StartZ']
    return np.sqrt(dx*dx + dy*dy + dz*dz)

# ==========================================================
# SECTION PARSING
# ==========================================================

def parse_section_dims(section_str):
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

# ==========================================================
# CUBOID
# ==========================================================

def create_cuboid(p1, p2, p3, p4, p5, p6, p7, p8):
    vertices = [p1,p2,p3,p4,p5,p6,p7,p8]
    indices = [
        0,2,1, 0,3,2,
        4,5,6, 4,6,7,
        0,1,5, 0,5,4,
        1,2,6, 1,6,5,
        2,3,7, 2,7,6,
        3,0,4, 3,4,7
    ]
    return vertices, indices

# ==========================================================
# LEG MESH
# ==========================================================

def generate_leg_mesh(row):
    start = np.array([row['StartX'], row['StartY'], row['StartZ']])
    end   = np.array([row['EndX'], row['EndY'], row['EndZ']])
    width, thickness = parse_section_dims(row['Section'])

    def inward(x): return -1.0 if x >= 0 else 1.0

    sx, sy = inward(start[0]), inward(start[1])
    ex, ey = inward(end[0]), inward(end[1])

    def box1(p, dx, dy):
        return (
            p,
            p + [width*dx,0,0],
            p + [width*dx,thickness*dy,0],
            p + [0,thickness*dy,0]
        )

    sb = box1(start,sx,sy)
    eb = box1(end,ex,ey)
    v1,i1 = create_cuboid(*sb,*eb)

    def box2(p, dx, dy):
        return (
            p,
            p + [thickness*dx,0,0],
            p + [thickness*dx,width*dy,0],
            p + [0,width*dy,0]
        )

    sb = box2(start,sx,sy)
    eb = box2(end,ex,ey)
    v2,i2 = create_cuboid(*sb,*eb)

    off = len(v1)
    return v1+v2, i1+[x+off for x in i2]

# ==========================================================
# GENERIC ORIENTED BOX
# ==========================================================

def get_oriented_box(row):
    start = np.array([row['StartX'], row['StartY'], row['StartZ']])
    end   = np.array([row['EndX'], row['EndY'], row['EndZ']])
    width,_ = parse_section_dims(row['Section'])
    thickness = width * 0.2

    v_long = end - start
    L = np.linalg.norm(v_long)
    if L == 0:
        return [], []
    v_long /= L

    mid = (start + end) / 2
    radial = np.array([mid[0], mid[1], 0])
    radial = unit(radial if np.linalg.norm(radial) else np.array([1,0,0]))

    v_width = unit(np.cross(v_long, radial))
    v_thick = unit(np.cross(v_long, v_width))

    w2 = width/2
    t2 = thickness/2

    corners = [
        -v_width*w2 - v_thick*t2,
         v_width*w2 - v_thick*t2,
         v_width*w2 + v_thick*t2,
        -v_width*w2 + v_thick*t2
    ]

    s = [start+c for c in corners]
    e = [p+(end-start) for p in s]

    return create_cuboid(*s,*e)

# ==========================================================
# FACE + HOLE GEOMETRY
# ==========================================================

def face_normal(leg1, leg2):
    v_leg = unit(vec(leg1))
    v_between = midpoint(leg2) - midpoint(leg1)
    return unit(np.cross(v_leg, v_between))

def offset_leg_line(leg, other_leg, leg_width):
    p0 = midpoint(leg)
    v_leg = unit(vec(leg))

    toward_center = unit(midpoint(other_leg) - p0)
    offset_dist = leg_width - HOLE_OFFSET_FACTOR * HOLE_DIAMETER

    return p0 + toward_center * offset_dist, v_leg

def closest_point_between_lines(p1, d1, p2, d2):
    d1 = unit(d1)
    d2 = unit(d2)
    r = p1 - p2

    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, r)
    e = np.dot(d2, r)

    denom = a * c - b * b
    if abs(denom) < 1e-9:
        return None

    t = (b * e - c * d) / denom
    return p1 + t * d1

def generate_cylinder(center, axis, radius, height, segments=24):
    axis = unit(axis)

    ref = np.array([0,0,1]) if abs(axis[2]) < 0.9 else np.array([1,0,0])
    v1 = unit(np.cross(axis, ref))
    v2 = np.cross(axis, v1)

    bottom = []
    top = []

    for a in np.linspace(0, 2*np.pi, segments, endpoint=False):
        offset = np.cos(a)*v1*radius + np.sin(a)*v2*radius
        bottom.append(center - axis*height/2 + offset)
        top.append(center + axis*height/2 + offset)

    verts = bottom + top
    idx = []

    for i in range(segments):
        j = (i + 1) % segments
        idx += [i, j, i+segments]
        idx += [j, j+segments, i+segments]

    return verts, idx

# ==========================================================
# HOLE DATA SAVING
# ==========================================================

def save_holes_to_csv(holes_data, csv_path="holes_data.csv"):
    """Save hole information to CSV file"""
    
    if not holes_data:
        print("No holes data to save")
        return
    
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "MemberID", "MemberType", "Section", 
            "HoleNumber", "HoleCenterX", "HoleCenterY", "HoleCenterZ",
            "NormalX", "NormalY", "NormalZ",
            "ConnectedLegID", "Radius", "Depth"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(holes_data)
    
    print(f"✓ Saved {len(holes_data)} holes to {csv_path}")

def merge_holes_with_members(csv_path, holes_csv="holes_data.csv", output_csv="data_with_holes.csv"):
    """Merge hole information back into member data"""
    
    # Read original data
    df = pd.read_csv(csv_path)
    
    # Read holes data
    try:
        holes_df = pd.read_csv(holes_csv)
    except FileNotFoundError:
        print(f"Error: {holes_csv} not found. Run visualize_tower first.")
        return None
    
    # Group holes by member
    holes_grouped = holes_df.groupby('MemberID').apply(
        lambda x: x.to_dict('records')
    ).to_dict()
    
    # Add holes information as JSON string
    df['Holes'] = df['ID'].apply(
        lambda x: str(holes_grouped.get(int(x), []))
    )
    
    # Also add hole count
    df['HoleCount'] = df['ID'].apply(
        lambda x: len(holes_grouped.get(int(x), []))
    )
    
    # Save merged data
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved merged data to {output_csv}")
    
    return df

# ==========================================================
# NEW: BRACE ANALYSIS BY SECTION
# ==========================================================

def analyze_braces_by_section(csv_path, holes_csv="holes_data.csv", output_csv="braces_analysis.csv"):
    """
    Analyze braces grouped by section with hole distance information
    
    Output columns:
    - Section: The section type (e.g., L40x4, L50x5)
    - BraceID: Member ID
    - Length: Total length of the brace
    - Hole1_DistFromStart: Distance from start point to first hole
    - Hole2_DistFromEnd: Distance from end point to second hole
    """
    
    # Read member data
    df = pd.read_csv(csv_path)
    df["MemberClass"] = df.apply(classify_member, axis=1)
    
    # Filter only braces
    braces = df[df.MemberClass == 'brace'].copy()
    
    # Read holes data
    try:
        holes_df = pd.read_csv(holes_csv)
    except FileNotFoundError:
        print(f"Error: {holes_csv} not found. Run visualize_tower first.")
        return None
    
    # Calculate brace lengths
    braces['Length'] = braces.apply(calculate_length, axis=1)
    
    # Initialize result list
    results = []
    
    print(f"\nAnalyzing {len(braces)} braces...")
    
    for _, brace in braces.iterrows():
        brace_id = int(brace['ID'])
        
        # Get holes for this brace
        brace_holes = holes_df[holes_df['MemberID'] == brace_id].copy()
        
        if len(brace_holes) == 0:
            # No holes found
            results.append({
                'Section': brace['Section'],
                'BraceID': brace_id,
                'Length': round(brace['Length'], 4),
                'Hole1_DistFromStart': None,
                'Hole2_DistFromEnd': None
            })
            continue
        
        # Brace start and end points
        start_point = np.array([brace['StartX'], brace['StartY'], brace['StartZ']])
        end_point = np.array([brace['EndX'], brace['EndY'], brace['EndZ']])
        
        # Calculate distances from start to each hole
        hole_distances = []
        for _, hole in brace_holes.iterrows():
            hole_center = np.array([hole['HoleCenterX'], hole['HoleCenterY'], hole['HoleCenterZ']])
            dist_from_start = np.linalg.norm(hole_center - start_point)
            dist_from_end = np.linalg.norm(hole_center - end_point)
            hole_distances.append({
                'hole_num': hole['HoleNumber'],
                'dist_from_start': dist_from_start,
                'dist_from_end': dist_from_end
            })
        
        # Sort by distance from start
        hole_distances.sort(key=lambda x: x['dist_from_start'])
        
        # Get first and last holes
        if len(hole_distances) >= 1:
            hole1_dist = hole_distances[0]['dist_from_start']
        else:
            hole1_dist = None
            
        if len(hole_distances) >= 2:
            hole2_dist = hole_distances[-1]['dist_from_end']
        elif len(hole_distances) == 1:
            hole2_dist = hole_distances[0]['dist_from_end']
        else:
            hole2_dist = None
        
        results.append({
            'Section': brace['Section'],
            'BraceID': brace_id,
            'Length': round(brace['Length'], 4),
            'Hole1_DistFromStart': round(hole1_dist, 4) if hole1_dist is not None else None,
            'Hole2_DistFromEnd': round(hole2_dist, 4) if hole2_dist is not None else None
        })
    
    # Convert to DataFrame and sort by Section
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['Section', 'BraceID'])
    
    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"✓ Saved brace analysis to {output_csv}")
    print(f"\nSummary by Section:")
    print(results_df.groupby('Section').size())
    
    return results_df


def generate_brace_bom_summary(
    input_csv="braces_analysis.csv",
    output_csv="braces_summary.csv"
):
    """
    Generate a Bill of Materials (BOM) style summary for braces.
    Groups identical braces by:
        - Section
        - Length
        - Hole distances

    Output:
        braces_summary.csv
    """

    import pandas as pd

    print("\nGenerating brace BOM summary...")

    df = pd.read_csv(input_csv)

    # Round to avoid floating point noise
    df["Length"] = df["Length"].round(3)
    df["Hole1_DistFromStart"] = df["Hole1_DistFromStart"].round(3)
    df["Hole2_DistFromEnd"] = df["Hole2_DistFromEnd"].round(3)

    summary = (
        df.groupby(
            ["Section", "Length", "Hole1_DistFromStart", "Hole2_DistFromEnd"]
        )
        .size()
        .reset_index(name="Quantity")
        .sort_values(["Section", "Quantity"], ascending=[True, False])
    )

    # Add a simple PartID for fabrication reference
    summary.insert(0, "PartID", range(1, len(summary) + 1))

    summary.to_csv(output_csv, index=False)

    print(f"✓ Brace BOM written to: {output_csv}")
    print(f"✓ Unique brace types: {len(summary)}")
    print(f"✓ Total braces: {summary['Quantity'].sum()}")

    return summary
# --------------------------------------------------
# FINAL STEP: Generate brace BOM summary
# --------------------------------------------------
generate_brace_bom_summary(
    input_csv="braces_analysis.csv",
    output_csv="braces_summary.csv"
)

# ==========================================================
# MAIN VISUALIZER
# ==========================================================

def visualize_tower(csv_path, save_holes=True):
    df = pd.read_csv(csv_path)
    df["MemberClass"] = df.apply(classify_member, axis=1)

    fig = go.Figure()
    holes_data = []  # Store hole information

    # ----- SOLIDS -----
    for cls,color in [('leg','red'),('brace','#00FF00'),('horizontal','blue')]:
        xs,ys,zs,i,j,k,hover_ids = [],[],[],[],[],[],[]
        off = 0
        for _,row in df[df.MemberClass==cls].iterrows():
            if cls=='leg':
                verts,idxs = generate_leg_mesh(row)
            else:
                verts,idxs = get_oriented_box(row)
            for v in verts:
                xs.append(v[0]); ys.append(v[1]); zs.append(v[2]); hover_ids.append(row.loc["ID"]) 
            for t in range(0,len(idxs),3):
                i.append(idxs[t]+off)
                j.append(idxs[t+1]+off)
                k.append(idxs[t+2]+off)
            off+=len(verts)
        fig.add_trace(go.Mesh3d(
            x=xs,y=ys,z=zs,i=i,j=j,k=k,
            color=color,opacity=1,
            customdata=hover_ids,
            hovertemplate=(
                "<b>Member ID:</b> %{customdata}<br>"
                "<b>Type:</b> " + cls + "<br>"
                "<extra></extra>"
            )
        ))

    # ----- HOLES ON BRACES + HORIZONTALS -----
    legs = df[df.MemberClass=='leg']
    face_members = df[df.MemberClass.isin(['brace','horizontal'])]

    print(f"Processing {len(face_members)} braces/horizontals for holes...")

    for idx, member in face_members.iterrows():
        p_start = np.array([member.loc['StartX'], member.loc['StartY'], member.loc['StartZ']])
        p_end   = np.array([member.loc['EndX'],   member.loc['EndY'],   member.loc['EndZ']])

        dists_start = [
            (np.linalg.norm(p_start - midpoint(l)), l)
            for _, l in legs.iterrows()
            if min(l.loc['StartZ'], l.loc['EndZ']) <= member.loc['StartZ'] <= max(l.loc['StartZ'], l.loc['EndZ'])
        ]

        dists_end = [
            (np.linalg.norm(p_end - midpoint(l)), l)
            for _, l in legs.iterrows()
            if min(l.loc['StartZ'], l.loc['EndZ']) <= member.loc['EndZ'] <= max(l.loc['StartZ'], l.loc['EndZ'])
        ]

        if not dists_start or not dists_end:
            print(f"⚠ Warning: No matching legs found for member {member.loc['ID']}")
            continue

        dists_start.sort(key=lambda x: x[0])
        dists_end.sort(key=lambda x: x[0])
        
        leg1 = dists_start[0][1]
        leg2 = dists_end[0][1]

        # Check if the end of the brace exactly on the middle point
        if leg2.loc['ID'] == leg1.loc['ID']:
            if len(dists_start) > 1 and dists_start[0][0] == dists_start[1][0]:
                leg1 = dists_start[1][1]
            elif len(dists_end) > 1:
                leg2 = dists_end[1][1]
        
        n_face = face_normal(leg1, leg2)
        
        hole_num = 1
        for A, B in [(leg1, leg2), (leg2, leg1)]:
            leg_width, _ = parse_section_dims(A['Section'])
            p_line, d_line = offset_leg_line(A, B, leg_width)

            member_p = np.array([member.StartX, member.StartY, member.StartZ])
            member_d = vec(member)

            hole_center = closest_point_between_lines(
                p_line, d_line, member_p, member_d
            )
            if hole_center is None:
                continue

            # Save hole data
            hole_info = {
                "MemberID": int(member.loc['ID']),
                "MemberType": member.loc['MemberClass'],
                "Section": member.loc['Section'],
                "HoleNumber": hole_num,
                "HoleCenterX": float(hole_center[0]),
                "HoleCenterY": float(hole_center[1]),
                "HoleCenterZ": float(hole_center[2]),
                "NormalX": float(n_face[0]),
                "NormalY": float(n_face[1]),
                "NormalZ": float(n_face[2]),
                "ConnectedLegID": int(A.loc['ID']),
                "Radius": HOLE_RADIUS,
                "Depth": HOLE_CYLINDER_LENGTH
            }
            holes_data.append(hole_info)
            hole_num += 1

            verts, idx = generate_cylinder(
                hole_center, n_face, HOLE_RADIUS, HOLE_CYLINDER_LENGTH
            )

            xs, ys, zs = zip(*verts)
            fig.add_trace(go.Mesh3d(
                x=xs, y=ys, z=zs,
                i=idx[0::3], j=idx[1::3], k=idx[2::3],
                color='black', opacity=1,
                showlegend=False
            ))

    # Save holes to CSV
    if save_holes and holes_data:
        save_holes_to_csv(holes_data, "holes_data.csv")

    fig.update_layout(
        scene=dict(aspectmode='data'),
        width=1200, height=900,
        title="BRK Visualizer – Holes on Braces & Horizontals"
    )
    fig.show()
    
    return holes_data

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    # Run visualization and hole detection
    holes = visualize_tower("data.csv", save_holes=True)
    print(f"\nTotal holes found: {len(holes)}")
    
    # Merge holes with members
    merge_holes_with_members("data.csv", "holes_data.csv", "data_with_holes.csv")
    
    # NEW: Analyze braces by section
    print("\n" + "="*50)
    print("Analyzing braces by section...")
    print("="*50)
    analyze_braces_by_section("data.csv", "holes_data.csv", "braces_analysis.csv")
