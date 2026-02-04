import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re

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
# LEG MESH (UNCHANGED)
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
# FACE + HOLE GEOMETRY (UNCHANGED LOGIC)
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
# MAIN VISUALIZER
# ==========================================================

def visualize_tower(csv_path):
    df = pd.read_csv(csv_path)
    df["MemberClass"] = df.apply(classify_member, axis=1)

    fig = go.Figure()

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
    face_members.to_csv("12.csv")

    for _,member in face_members.iterrows():
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
            raise ValueError("No matching legs found for brace")

        dists_start.sort(key=lambda x: x[0])
        dists_end.sort(key=lambda x: x[0])
        
        leg1 = dists_start[0][1]
        leg2 = dists_end[0][1]


        # check if the end of the brace exacly on the middle point
        if leg2.loc['ID'] == leg1.loc['ID']:
            if dists_start[0][0] == dists_start[1][0]:
                leg1 = dists_start[1][1]
            else:
                leg2 = dists_end[1][1]
        n_face = face_normal(leg1,leg2)
        #draw normal face of each brace
        nx, ny, nz = [], [], []
        mid = midpoint(member)
        vec_face_x = n_face[0] + mid[0]
        vec_face_y = n_face[1] + mid[1]
        vec_face_z = n_face[2] + mid[2]
        nx.extend([mid[0], vec_face_x, None])
        ny.extend([mid[1], vec_face_y, None])
        nz.extend([mid[2], vec_face_z, None])
        fig.add_trace(go.Scatter3d(
            x=nx, y=ny, z=nz,
            mode='lines',
            line=dict(color='yellow', width=6),
            name='Brace n_face'
        ))
        for A,B in [(leg1,leg2),(leg2,leg1)]:
            leg_width,_ = parse_section_dims(A['Section'])
            p_line,d_line = offset_leg_line(A,B,leg_width)

            member_p = np.array([member.StartX,member.StartY,member.StartZ])
            member_d = vec(member)

            hole_center = closest_point_between_lines(
                p_line,d_line,member_p,member_d
            )
            if hole_center is None:
                continue

            verts,idx = generate_cylinder(
                hole_center,n_face,HOLE_RADIUS,HOLE_CYLINDER_LENGTH
            )

            xs,ys,zs = zip(*verts)
            fig.add_trace(go.Mesh3d(
                x=xs,y=ys,z=zs,
                i=idx[0::3],j=idx[1::3],k=idx[2::3],
                color='black',opacity=1,
            ))

    fig.update_layout(
        scene=dict(aspectmode='cube'),
        width=1200,height=900,
        title="BRK Visualizer – Holes on Braces & Horizontals"
    )
    fig.show()

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    visualize_tower("data.csv")
