========
# SET YOUR PATHS HERE
# ============================================================
IMAGE_PATH = r"C:\Users\syedw\OneDrive\Desktop\uas\input"
OUTPUT_DIR = r"C:\Users\syedw\OneDrive\Desktop\uas\output"
# ============================================================

import heapq
import json
import os

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

AGE_SCORE = {"circle": 3, "star": 1, "square": 2}
AGE_NAME = {"circle": "Children", "star": "Adults", "square": "Senior Citizens"}
SEVERITY_SCORE = {"red": 3, "yellow": 2, "white": 1}
SEVERITY_NAME = {"red": "Critical", "yellow": "Moderate", "white": "Safe"}

SPEED_BY_LEVEL = {0: 20.0, 1: 15.0, 2: 10.0}
DEFAULT_SPEED = 10.0

GRID_SCALE = 4
MIN_SHAPE_AREA = 40
MIN_MARKER_AREA = 150
NUM_TERRAIN_LEVELS = 3  # light / moderately dark / darkest green


def shape_from_contour(cnt):
    peri = cv2.arcLength(cnt, True)
    if peri <= 0:
        return "unknown"
    approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)
    area = cv2.contourArea(cnt)
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
    vertices = len(approx)

    if vertices == 3:
        return "triangle"
    if circularity > 0.80:
        return "circle"
    if vertices == 4:
        return "square"
    return "star"


def mean_hsv_of_contour(hsv_img, cnt):
    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    mean = cv2.mean(hsv_img, mask=mask)
    return mean[0], mean[1], mean[2]


def classify_casualty_color(h, s, v):
    if s < 50 and v > 140:
        return "white"
    if (h <= 10 or h >= 170) and s > 60:
        return "red"
    if 18 <= h <= 40 and s > 60:
        return "yellow"
    return None


def build_masks(img_bgr):
    """
    Build the obstacle mask, casualty-colour masks and a terrain elevation
    level mask.

    Terrain levels are found with k-means clustering on pixel brightness
    (capped at NUM_TERRAIN_LEVELS clusters), NOT by counting exact unique
    RGB values. Counting unique colours is fragile on real / JPEG /
    anti-aliased images because compression noise can create hundreds of
    near-identical shades of green, which previously blew past the int8
    range used for level indices. K-means keeps the level count bounded
    and robust regardless of image noise.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    obstacle_mask = ((v < 60)).astype(np.uint8) * 255
    green_mask = (((h >= 32) & (h <= 95)) & (s > 40)).astype(np.uint8) * 255
    purple_mask = (((h >= 110) & (h <= 165)) & (s > 40)).astype(np.uint8) * 255
    orange_mask = (((h >= 8) & (h <= 18)) & (s > 90) & (v > 90)).astype(np.uint8) * 255
    red_mask = ((((h <= 10) | (h >= 170))) & (s > 60) & (v > 60)).astype(np.uint8) * 255
    yellow_mask = (((h >= 18) & (h <= 40)) & (s > 60) & (v > 90)).astype(np.uint8) * 255
    white_mask = ((s < 50) & (v > 140)).astype(np.uint8) * 255

    # int16 (not int8) gives extra headroom for the level index.
    level_mask = np.full(img_bgr.shape[:2], -1, dtype=np.int16)
    ys, xs = np.where(green_mask > 0)
    if len(ys) > 0:
        pixel_vals = img_bgr[ys, xs].astype(np.float32)
        brightness = pixel_vals.sum(axis=1).reshape(-1, 1)  # 1D clustering feature

        n_unique = len(np.unique(np.round(brightness).astype(np.int32)))
        k_eff = max(1, min(NUM_TERRAIN_LEVELS, n_unique))

        if k_eff == 1:
            labels_flat = np.zeros(len(brightness), dtype=np.int32)
            centers = np.array([[brightness.mean()]], dtype=np.float32)
        else:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
            _, labels, centers = cv2.kmeans(
                brightness, k_eff, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            labels_flat = labels.flatten()

        centers_flat = centers.flatten()
        order = np.argsort(centers_flat)[::-1]  # brightest (lightest green) -> level 0
        rank = {int(old): new for new, old in enumerate(order)}
        levels_flat = np.array([rank[int(l)] for l in labels_flat], dtype=np.int16)
        level_mask[ys, xs] = levels_flat

    return obstacle_mask, level_mask, purple_mask, orange_mask, red_mask, yellow_mask, white_mask


def fill_level_holes(level_mask, obstacle_mask):
    known = level_mask >= 0
    if known.all():
        filled = level_mask.copy()
    else:
        if not known.any():
            filled = np.zeros_like(level_mask)
        else:
            _, inds = distance_transform_edt(~known, return_distances=True, return_indices=True)
            filled = level_mask[tuple(inds)]
    filled = filled.copy()
    filled[obstacle_mask > 0] = -1
    return filled


def detect_markers_and_casualties(img_bgr, obstacle_mask, level_mask, purple_mask,
                                   orange_mask, red_mask, yellow_mask, white_mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = obstacle_mask.shape

    start_pos = None
    best_area = 0
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_MARKER_AREA:
            continue
        if shape_from_contour(cnt) == "triangle" and area > best_area:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            start_pos = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            best_area = area

    destination_pos = None
    best_area = 0
    contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_MARKER_AREA:
            continue
        if area > best_area:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            destination_pos = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            best_area = area

    casualty_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, yellow_mask), white_mask)
    contours, _ = cv2.findContours(casualty_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    casualties = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_SHAPE_AREA:
            continue
        shp = shape_from_contour(cnt)
        if shp not in AGE_SCORE:
            continue
        mh, ms, mv = mean_hsv_of_contour(hsv, cnt)
        color = classify_casualty_color(mh, ms, mv)
        if color is None:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        lvl = int(level_mask[min(max(cy, 0), h - 1), min(max(cx, 0), w - 1)])
        lvl = max(lvl, 0)
        priority = SEVERITY_SCORE[color] * AGE_SCORE[shp]
        casualties.append({
            "coords": (cx, cy), "shape": shp, "age_group": AGE_NAME[shp],
            "age_score": AGE_SCORE[shp], "color": color, "severity": SEVERITY_NAME[color],
            "severity_score": SEVERITY_SCORE[color], "priority_score": priority, "level": lvl,
        })

    return start_pos, destination_pos, casualties


def downsample_grid(obstacle_mask, level_mask, scale):
    h, w = obstacle_mask.shape
    h2, w2 = (h // scale) * scale, (w // scale) * scale
    obs = obstacle_mask[:h2, :w2].reshape(h2 // scale, scale, w2 // scale, scale)
    lvl = level_mask[:h2, :w2].reshape(h2 // scale, scale, w2 // scale, scale)
    grid_obstacle = obs.mean(axis=(1, 3)) > 127
    grid_level = np.round(lvl.astype(np.float32).mean(axis=(1, 3))).astype(np.int16)
    grid_level[grid_level < 0] = 0
    return grid_obstacle, grid_level


def to_grid_coord(px, py, scale, gh, gw):
    gx = min(max(px // scale, 0), gw - 1)
    gy = min(max(py // scale, 0), gh - 1)
    return gy, gx


def dijkstra(grid_obstacle, start, goal):
    gh, gw = grid_obstacle.shape
    dist = np.full((gh, gw), np.inf, dtype=np.float64)
    prev = {}
    sy, sx = start
    if grid_obstacle[sy, sx]:
        grid_obstacle = grid_obstacle.copy()
        grid_obstacle[sy, sx] = False
    dist[sy, sx] = 0.0
    heap = [(0.0, (sy, sx))]
    visited = set()
    neighbours = [(-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
                  (0, -1, 1.0), (0, 1, 1.0),
                  (1, -1, np.sqrt(2)), (1, 0, 1.0), (1, 1, np.sqrt(2))]
    goal_reached = False
    while heap:
        d, (y, x) = heapq.heappop(heap)
        if (y, x) in visited:
            continue
        visited.add((y, x))
        if (y, x) == goal:
            goal_reached = True
            break
        for dy, dx, base in neighbours:
            ny, nx = y + dy, x + dx
            if 0 <= ny < gh and 0 <= nx < gw and not grid_obstacle[ny, nx]:
                nd = d + base
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[(ny, nx)] = (y, x)
                    heapq.heappush(heap, (nd, (ny, nx)))
    if not goal_reached or dist[goal] == np.inf:
        return None, np.inf
    path = [goal]
    cur = goal
    while cur != (sy, sx):
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path, dist[goal]


def grid_path_to_pixels_and_time(path_grid, grid_level, scale):
    pts = [(gx * scale + scale // 2, gy * scale + scale // 2) for gy, gx in path_grid]
    total_dist = 0.0
    total_time = 0.0
    per_level_dist = {}
    for i in range(1, len(pts)):
        x1, y1 = pts[i - 1]
        x2, y2 = pts[i]
        d = float(np.hypot(x2 - x1, y2 - y1))
        gy, gx = path_grid[i]
        lvl = int(grid_level[gy, gx])
        speed = SPEED_BY_LEVEL.get(lvl, DEFAULT_SPEED)
        total_dist += d
        total_time += d / speed
        per_level_dist[lvl] = per_level_dist.get(lvl, 0.0) + d
    return pts, total_dist, total_time, per_level_dist


def shortest_path_pixels(grid_obstacle, grid_level, scale, p_from, p_to):
    gh, gw = grid_obstacle.shape
    s = to_grid_coord(p_from[0], p_from[1], scale, gh, gw)
    g = to_grid_coord(p_to[0], p_to[1], scale, gh, gw)
    path_grid, grid_dist = dijkstra(grid_obstacle, s, g)
    if path_grid is None:
        return None, np.inf, np.inf, {}
    pts, dist, time_, per_level = grid_path_to_pixels_and_time(path_grid, grid_level, scale)
    return pts, dist, time_, per_level


def plan_route(grid_obstacle, grid_level, scale, start_pos, destination_pos, casualties):
    remaining = list(range(len(casualties)))
    current_pos = start_pos
    cum_dist = 0.0
    full_path = [start_pos]
    ordered_results = []

    while remaining:
        best_idx = None
        best_score = -np.inf
        best_pts = None
        best_leg_dist = None

        for idx in remaining:
            c = casualties[idx]
            pts, leg_dist, leg_time, _ = shortest_path_pixels(
                grid_obstacle, grid_level, scale, current_pos, c["coords"])
            if pts is None:
                continue
            projected_cum = cum_dist + leg_dist
            displacement = float(np.hypot(c["coords"][0] - start_pos[0],
                                           c["coords"][1] - start_pos[1]))
            score = (displacement / projected_cum) * c["priority_score"] if projected_cum > 0 else 0.0
            if score > best_score:
                best_score = score
                best_idx = idx
                best_pts = pts
                best_leg_dist = leg_dist

        if best_idx is None:
            break

        remaining.remove(best_idx)
        c = casualties[best_idx]
        cum_dist += best_leg_dist
        current_pos = c["coords"]
        full_path.extend(best_pts[1:])

        displacement = float(np.hypot(c["coords"][0] - start_pos[0],
                                       c["coords"][1] - start_pos[1]))
        casualty_score = (displacement / cum_dist) * c["priority_score"] if cum_dist > 0 else 0.0

        result = dict(c)
        result["displacement_from_start"] = round(displacement, 2)
        result["distance_travelled_to_reach"] = round(cum_dist, 2)
        result["casualty_score"] = round(casualty_score, 4)
        ordered_results.append(result)

    pts, leg_dist, leg_time, _ = shortest_path_pixels(
        grid_obstacle, grid_level, scale, current_pos, destination_pos)
    if pts is not None:
        full_path.extend(pts[1:])

    return ordered_results, full_path


def compute_total_time(full_path, level_mask):
    total_time = 0.0
    per_level_time = {}
    h, w = level_mask.shape
    for i in range(1, len(full_path)):
        x1, y1 = full_path[i - 1]
        x2, y2 = full_path[i]
        d = float(np.hypot(x2 - x1, y2 - y1))
        yy = min(max(y2, 0), h - 1)
        xx = min(max(x2, 0), w - 1)
        lvl = max(int(level_mask[yy, xx]), 0)
        speed = SPEED_BY_LEVEL.get(lvl, DEFAULT_SPEED)
        t = d / speed
        total_time += t
        per_level_time[lvl] = per_level_time.get(lvl, 0.0) + t
    return total_time, per_level_time


def save_mask_image(obstacle_mask, out_path):
    mask_img = np.full((*obstacle_mask.shape, 3), 255, dtype=np.uint8)
    mask_img[obstacle_mask > 0] = (0, 0, 0)
    cv2.imwrite(out_path, mask_img)


def save_path_visualisation(img_bgr, start_pos, destination_pos, casualties_ordered,
                             full_path, out_path):
    vis = img_bgr.copy()
    pts = np.array(full_path, dtype=np.int32)
    for i in range(1, len(pts)):
        cv2.line(vis, tuple(pts[i - 1]), tuple(pts[i]), (0, 0, 0), 2)
    if start_pos:
        cv2.circle(vis, start_pos, 8, (0, 140, 255), -1)
    if destination_pos:
        cv2.circle(vis, destination_pos, 8, (255, 0, 255), -1)
    for c in casualties_ordered:
        cv2.circle(vis, c["coords"], 6, (0, 0, 0), 2)
    cv2.imwrite(out_path, vis)


def process_image(img_path, output_dir):
    name = os.path.splitext(os.path.basename(img_path))[0]
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    obstacle_mask, level_mask, purple_mask, orange_mask, red_mask, yellow_mask, white_mask = build_masks(img_bgr)
    level_mask = fill_level_holes(level_mask, obstacle_mask)

    start_pos, destination_pos, casualties = detect_markers_and_casualties(
        img_bgr, obstacle_mask, level_mask, purple_mask, orange_mask, red_mask, yellow_mask, white_mask)

    mask_out = os.path.join(output_dir, f"{name}_mask.png")
    save_mask_image(obstacle_mask, mask_out)

    if start_pos is None or destination_pos is None:
        report = {"image": name, "error": "start or destination marker not found",
                  "num_casualties": len(casualties), "casualties": casualties}
        with open(os.path.join(output_dir, f"{name}_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        return report