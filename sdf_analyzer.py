import csv
import re
from tower_visualizer import visualize_tower, merge_holes_with_members
import pandas as pd
import numpy as np

def parse_sdf_to_csv(sdf_path, csv_path="data.csv"):
    element_start = re.compile(r"^(\d{8})\s+\d+\s+\d+\s+\d+\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+(\d+)")
    elements = []
    with open(sdf_path, "r", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        m = element_start.match(line)
        if not m:
            i += 1
            continue

        member_id = m.group(1)
        member_type = m.group(2)
        piece_mark = m.group(3)

        if i + 5 >= n:
            break

        line2 = lines[i+1].strip()
        line3 = lines[i+2].strip()

        sec_match = re.match(r"\"([^\"]+)\"\s+\"([^\"]+)\"", line2)
        if sec_match:
            section = sec_match.group(1)
            material = sec_match.group(2)
        else:
            section = ""
            material = ""

        nums = re.findall(r"[-+]?\d*\.\d+|\d+", line3)

        if len(nums) >= 9:
            sx, sy, sz = map(float, nums[3:6])
            ex, ey, ez = map(float, nums[6:9])
        else:
            sx = sy = sz = ex = ey = ez = None

        elements.append({
            "ID": member_id,
            "Type": member_type,
            "PieceMark": piece_mark,
            "Section": section,
            "Material": material,
            "StartX": sx,
            "StartY": sy,
            "StartZ": sz,
            "EndX": ex,
            "EndY": ey,
            "EndZ": ez
        })

        i += 6  

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=elements[0].keys())
        writer.writeheader()
        writer.writerows(elements)

    print(f"Saved {len(elements)} elements to {csv_path}")
    return elements

def classify_members(elements):
    legs = []
    braces = []
    horizontals = []

    for e in elements:
        dx = e["EndX"] - e["StartX"]
        dy = e["EndY"] - e["StartY"]
        dz = e["EndZ"] - e["StartZ"]

        length = np.sqrt(dx*dx + dy*dy + dz*dz)
        if length == 0:
            continue

        angle = np.degrees(np.arccos(abs(dz) / length))

        # Classification
        if angle < 10:            
            legs.append(e)
        elif angle > 80:          
            horizontals.append(e)
        else:                     
            braces.append(e)

    return legs, braces, horizontals

def assaing_brace_to_leg(braces, legs):
    def midpoint(member):
        return np.array([
            (member['StartX'] + member['EndX']) / 2,
            (member['StartY'] + member['EndY']) / 2,
            (member['StartZ'] + member['EndZ']) / 2
        ])

    for brace in braces:
        brace['Leg_Start'] = None
        brace['Leg_End'] = None

        p_start = np.array([brace['StartX'], brace['StartY'], brace['StartZ']])
        p_end   = np.array([brace['EndX'],   brace['EndY'],   brace['EndZ']])

        # ---------- START POINT ----------
        dists_start = [
            (np.linalg.norm(p_start - midpoint(leg)), leg)
            for leg in legs
            if min(leg['StartZ'], leg['EndZ']) <= midpoint(brace)[2] <= max(leg['StartZ'], leg['EndZ'])
        ]

        # ---------- END POINT ----------
        dists_end = [
            (np.linalg.norm(p_end - midpoint(leg)), leg)
            for leg in legs
            if min(leg['StartZ'], leg['EndZ']) <= midpoint(brace)[2] <= max(leg['StartZ'], leg['EndZ'])
        ]

        if not dists_start or not dists_end:
            raise ValueError(f"No matching legs found for brace ID {brace['ID']}")

        dists_start.sort(key=lambda x: x[0])
        dists_end.sort(key=lambda x: x[0])

        brace['Leg_Start'] = dists_start[0][1]
        brace['Leg_End']   = dists_end[0][1]
        if brace['Leg_Start']['ID'] == brace['Leg_End']['ID']:
            if dists_start[0][0] == dists_start[1][0]:
                brace['Leg_Start'] = dists_start[1][1]
            else:
                brace['Leg_End'] = dists_end[1][1]

    return braces

def space_from_legs(braces, space):
    def length(vector):
        return  np.sqrt((vector['StartX'] - vector['EndX'])**2 + (vector['StartY'] - vector['EndY'])**2 + (vector['StartZ'] - vector['EndZ'])**2)
    def Unit_vector(vector):
        distance = length(vector)
        return ((vector['StartX'] - vector['EndX'])/distance, (vector['StartY'] - vector['EndY'])/distance, (vector['StartZ'] - vector['EndZ'])/distance)
    for brace in braces:
        angle1 = np.acos( length(Unit_vector(brace['Leg_Start'])) / length(Unit_vector(brace)))
        angle2 = np.acos( length(Unit_vector(brace['Leg_End'])) / length(Unit_vector(brace)))
        k1 = space / (np.sin(angle1) * distance(Unit_vector(vector)))
        k2 = space / (np.sin(angle2) * distance(Unit_vector(vector)))
        brace['StartX'] += k1 * Unit_vector(vector)['StartX']
        brace['StartY'] += k1 * Unit_vector(vector)['StartY']
        brace['StartZ'] += k1 * Unit_vector(vector)['StartZ']
        brace['EndX'] -= k1 * Unit_vector(vector)['EndX']
        brace['EndY'] -= k1 * Unit_vector(vector)['EndY']
        brace['EndZ'] -= k1 * Unit_vector(vector)['EndZ']
    return braces

def space_from_legs_2(braces, space):
    def vec(v):
        return np.array([v['EndX'] - v['StartX'], v['EndY'] - v['StartY'], v['EndZ'] - v['StartZ']])
    for brace in braces:
        v_brace = vec(brace)
        if brace['Leg_Start'] is not None :
            v_leg_start = vec(brace['Leg_Start'])
            angle1 = np.arccos(np.dot(v_brace, v_leg_start) / (np.linalg.norm(v_brace) * np.linalg.norm(v_leg_start)))
            k1 = space / np.sin(angle1)
            unit_brace = v_brace / np.linalg.norm(v_brace)
            brace['StartX'] += k1 * unit_brace[0]
            brace['StartY'] += k1 * unit_brace[1]
            brace['StartZ'] += k1 * unit_brace[2]
        
        if  brace['Leg_End'] is not None:
            v_leg_end = vec(brace['Leg_End'])
            angle2 = np.arccos(np.dot(v_brace, v_leg_end) / (np.linalg.norm(v_brace) * np.linalg.norm(v_leg_end)))
            k2 = space / np.sin(angle2)
            brace['EndX'] -= k2 * unit_brace[0]
            brace['EndY'] -= k2 * unit_brace[1]
            brace['EndZ'] -= k2 * unit_brace[2]
    return braces


def save_to_csv(elements, csv_path="data.csv"):
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=elements[0].keys())
        writer.writeheader()
        writer.writerows(elements)

def remove_assained_braces(braces):
    braces_without_leg = []
    for i in braces :
            i.pop('Leg_End')
            i.pop('Leg_Start')
            braces_without_leg.append(i)
            
    return braces_without_leg

def center_of_tower(legs):
    sum_x = 0
    sum_y = 0
    for leg in legs:
        sum_x += leg['StartX']
        sum_y += leg['StartY']
    return [sum_x / len(legs), sum_y / len(legs), 0]

def push_out_of_leg(braces, center_t, space):
    def vec(v):
        return np.array([v['EndX'] - v['StartX'], 
                         v['EndY'] - v['StartY'], 
                         v['EndZ'] - v['StartZ']])

    def vec_center(brace):
        return np.array([center_t[0] - brace['StartX'], 
                         center_t[1] - brace['StartY'], 
                         center_t[2] - brace['StartZ']])
    for brace in braces:
        v_brace = vec(brace)
        if brace['Leg_Start']:
            v_leg = vec(brace['Leg_Start'])
        else:
            v_leg = vec(brace['Leg_End'])
        cross_vec = np.cross(v_brace, v_leg)
        side = np.dot(cross_vec, vec_center(brace))
        if np.linalg.norm(cross_vec) == 0:
            continue
        dir_vec = cross_vec / np.linalg.norm(cross_vec)
        if side >= 0:
            dir_vec = -dir_vec
        brace['StartX'] += dir_vec[0] * space
        brace['EndX']   += dir_vec[0] * space
        brace['StartY'] += dir_vec[1] * space
        brace['EndY']   += dir_vec[1] * space
        brace['StartZ'] += dir_vec[2] * space
        brace['EndZ']   += dir_vec[2] * space

    return braces
        

if "__main__" == __name__:
    elements = parse_sdf_to_csv("11.sdf")
    legs, braces, horizontals = classify_members(elements)
    assainged_braces = assaing_brace_to_leg(braces, legs)
    save_to_csv(assainged_braces,"assainged_braaces.csv")
    spaced_braces = space_from_legs_2(assainged_braces, 0.1) 
    center_tower = center_of_tower(legs)
    pushed_out_braces = push_out_of_leg(assainged_braces, center_tower, 0)
    pushed_out_braces = push_out_of_leg(assainged_braces, center_tower, 0)
    save_to_csv(legs + horizontals + remove_assained_braces(assainged_braces))
    
    # Visualize and get holes
    print("\n" + "="*50)
    print("Starting visualization and hole detection...")
    print("="*50 + "\n")
    holes = visualize_tower("data.csv", save_holes=True)
    
    # Merge holes back into member data
    print("\n" + "="*50)
    print("Merging holes with member data...")
    print("="*50 + "\n")
    merge_holes_with_members("data.csv", "holes_data.csv", "data_with_holes.csv")
    
    print(f"\nTotal holes found: {len(holes)}")
    print("Processing complete!")
