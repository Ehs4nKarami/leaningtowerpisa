import csv
import re
from tower_visualizer import visualize_tower
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

def assaing_brace_to_leg(braces, legs, tol=0):
    for i in braces:
        i['Leg_Start'] = None
        i['Leg_End'] = None
        for j in legs:
            if (min(j['StartZ'], j['EndZ']) - tol <= i['StartZ'] <= max(j['StartZ'], j['EndZ']) + tol) \
               and (min(j['StartY'], j['EndY']) - tol <= i['StartY'] <= max(j['StartY'], j['EndY']) + tol) \
               and (min(j['StartX'], j['EndX']) - tol <= i['StartX'] <= max(j['StartX'], j['EndX']) + tol):
                i['Leg_Start'] = j

            if (min(j['StartZ'], j['EndZ']) - tol <= i['EndZ'] <= max(j['StartZ'], j['EndZ']) + tol) \
               and (min(j['StartY'], j['EndY']) - tol <= i['EndY'] <= max(j['StartY'], j['EndY']) + tol) \
               and (min(j['StartX'], j['EndX']) - tol <= i['EndX'] <= max(j['StartX'], j['EndX']) + tol):
                i['Leg_End'] = j

            if i['Leg_Start'] is not None and i['Leg_End'] is not None:
                break
        if i['Leg_Start'] is None or i['Leg_End'] is None:
            print(f"Warning: Brace ID {i['ID']} missing leg linkage (tol={tol})")

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
        v_leg_start = vec(brace['Leg_Start'])
        v_leg_end = vec(brace['Leg_End'])
        angle1 = np.arccos(np.dot(v_brace, v_leg_start) / (np.linalg.norm(v_brace) * np.linalg.norm(v_leg_start)))
        angle2 = np.arccos(np.dot(v_brace, v_leg_end) / (np.linalg.norm(v_brace) * np.linalg.norm(v_leg_end)))
        k1 = space / np.sin(angle1)
        k2 = space / np.sin(angle2)
        unit_brace = v_brace / np.linalg.norm(v_brace)
        brace['StartX'] += k1 * unit_brace[0]
        brace['StartY'] += k1 * unit_brace[1]
        brace['StartZ'] += k1 * unit_brace[2]
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
        if i['Leg_Start'] is  None or i['Leg_End'] is  None:
            i.pop('Leg_End')
            i.pop('Leg_Start')
            braces_without_leg.append(i)
            
    return braces_without_leg

if "__main__" == __name__:
    elements = parse_sdf_to_csv("11.sdf")
    legs, braces, horizontals = classify_members(elements)
    assainged_braces = assaing_brace_to_leg(braces, legs)
    save_to_csv(legs + horizontals + remove_assained_braces(assainged_braces))
#    spaced_braces = space_from_legs_2(assainged_braces, 4) 
#    save_to_csv(legs + spaced_braces + horizontals)
    visualize_tower("data.csv")

