import os
import re
import numpy as np
import math
from . import evaluator

# --- PHẦN 1: CÁC HÀM TÍNH TOÁN KHOẢNG CÁCH ---

def _calculate_euc_2d_matrix(coords):
    """Tạo ma trận khoảng cách EUC_2D từ list tọa độ (x, y)."""
    points = np.array(coords)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    distances = np.sqrt(dist_sq)
    return np.rint(distances).astype(int)

def _calculate_geo_matrix(coords):
    """Tạo ma trận khoảng cách GEO từ list tọa độ (lat, lon)."""
    n = len(coords)
    matrix = np.zeros((n, n), dtype=int)
    R = 6378.388
    
    rad_coords = []
    for lat_deg, lon_deg in coords:
        lat_rad = (math.pi * lat_deg / 180.0)
        lon_rad = (math.pi * lon_deg / 180.0)
        rad_coords.append((lat_rad, lon_rad))

    for i in range(n):
        for j in range(i + 1, n):
            lat_i, lon_i = rad_coords[i]
            lat_j, lon_j = rad_coords[j]
            
            q1 = math.cos(lon_i - lon_j)
            q2 = math.cos(lat_i - lat_j)
            q3 = math.cos(lat_i + lat_j)
            
            arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
            arg = min(1.0, max(-1.0, arg))
            
            dist_ij = R * math.acos(arg) + 1.0
            dist_int = int(np.rint(dist_ij))
            
            matrix[i, j] = dist_int
            matrix[j, i] = dist_int
            
    return matrix

def _calculate_att_matrix(coords):
    """Tạo ma trận khoảng cách ATT (pseudo-Euclidean)."""
    n = len(coords)
    matrix = np.zeros((n, n), dtype=int)
    points = np.array(coords)

    for i in range(n):
        for j in range(i + 1, n):
            xd = points[i, 0] - points[j, 0]
            yd = points[i, 1] - points[j, 1]
            
            r = math.sqrt((xd**2 + yd**2) / 10.0)
            dist_int = int(np.rint(r))
            
            matrix[i, j] = dist_int
            matrix[j, i] = dist_int
            
    return matrix

# --- PHẦN 2: CÁC HÀM PHÂN TÍCH (PARSING) ---

def _parse_coords_from_lines(coord_lines, dimension):
    """Phân tích NODE_COORD_SECTION."""
    coords = []
    for line in coord_lines:
        line = line.strip()
        if not line: continue
        try:
            parts = [float(x) for x in line.split()]
            if len(parts) >= 3:
                coords.append((int(parts[0]), parts[1], parts[2]))
            elif len(parts) == 2:
                coords.append((len(coords) + 1, parts[0], parts[1]))
        except ValueError:
            continue
            
    if len(coords) != dimension:
        print(f"Cảnh báo: DIMENSION={dimension} nhưng có {len(coords)} tọa độ.")
    
    coords.sort(key=lambda x: x[0]) 
    final_coords = [(c[1], c[2]) for c in coords]
    
    return final_coords

def _parse_explicit_matrix(matrix_lines, dimension, edge_weight_format):
    """Phân tích ma trận khoảng cách từ định dạng EXPLICIT."""
    data_str = ' '.join(matrix_lines)
    weights = []
    for x in data_str.split():
        try:
            weights.append(int(float(x)))
        except ValueError:
            continue

    matrix = np.zeros((dimension, dimension), dtype=int)
    k = 0 

    if edge_weight_format == 'FULL_MATRIX':
        for i in range(dimension):
            for j in range(dimension):
                if k < len(weights):
                    matrix[i, j] = weights[k]
                    k += 1
                    
    elif edge_weight_format == 'UPPER_ROW':
        for i in range(dimension):
            for j in range(i + 1, dimension):
                if k < len(weights):
                    matrix[i, j] = weights[k]
                    matrix[j, i] = weights[k]
                    k += 1
                    
    elif edge_weight_format == 'LOWER_ROW':
        for i in range(dimension):
            for j in range(0, i):
                if k < len(weights):
                    matrix[i, j] = weights[k]
                    matrix[j, i] = weights[k]
                    k += 1
                    
    elif edge_weight_format == 'LOWER_DIAG_ROW':
        for i in range(dimension):
            for j in range(i + 1): 
                if k < len(weights):
                    matrix[i, j] = weights[k]
                    matrix[j, i] = weights[k]
                    k += 1
    else:
        raise NotImplementedError(f"Format '{edge_weight_format}' chưa hỗ trợ.")

    return matrix

def _handle_outlier_weights(matrix, problem_name):
    """
    Xử lý outlier weights (như brg180 có edges = 10000).
    
    Chiến lược:
    - Tính median và percentile của weights
    - Nếu có weights >> median → coi như inf (không đi được)
    
    Args:
        matrix: Ma trận khoảng cách
        problem_name: Tên problem (để log)
    
    Returns:
        matrix đã được xử lý
    """
    # Lấy tất cả weights (trừ diagonal = 0)
    n = len(matrix)
    off_diag = matrix[~np.eye(n, dtype=bool)]
    
    if len(off_diag) == 0:
        return matrix
    
    # Tính statistics
    unique_weights = np.unique(off_diag)
    median = np.median(off_diag)
    p75 = np.percentile(off_diag, 75)  # 75th percentile
    p95 = np.percentile(off_diag, 95)  # 95th percentile
    max_weight = np.max(off_diag)
    
    # Threshold: nếu weight > 10x median HOẶC > 3x p95
    threshold_1 = median * 10
    threshold_2 = p95 * 3
    threshold = min(threshold_1, threshold_2)
    
    # Đặc biệt: với brg180, median=30, max=10000
    # threshold = min(300, 3*9000) = 300
    # → Weights 3500, 9000, 10000 sẽ bị set thành inf
    
    # Đếm outliers
    outlier_mask = matrix > threshold
    num_outliers = np.sum(outlier_mask)
    
    if num_outliers > 0:
        print(f"  📊 Weight statistics for {problem_name}:")
        print(f"     Unique weights: {unique_weights}")
        print(f"     Median: {median:.0f}, P75: {p75:.0f}, P95: {p95:.0f}, Max: {max_weight:.0f}")
        print(f"     Threshold: {threshold:.0f}")
        print(f"  ⚠️  Found {num_outliers} outlier edges (weight > {threshold:.0f})")
        
        # CRITICAL: Chỉ set inf nếu outliers chiếm < 50% edges
        # Nếu quá nhiều → có thể đây là bài toán đặc biệt
        total_edges = n * (n - 1)
        outlier_ratio = num_outliers / total_edges
        
        if outlier_ratio < 0.5:
            print(f"     Setting outliers to inf ({outlier_ratio*100:.1f}% of edges)")
            matrix = matrix.copy()
            matrix[outlier_mask] = float('inf')
        else:
            print(f"     ⚠️  Too many outliers ({outlier_ratio*100:.1f}%), keeping original weights")
            print(f"     This problem may have special structure")
    
    return matrix

# --- PHẦN 3: HÀM GIAO DIỆN CHÍNH ---

def load_tsp_problem(problem_name, data_dir, handle_outliers=True):
    """
    Tải bài toán TSP từ tên file.
    
    Args:
        problem_name: Tên file (có thể có hoặc không có .tsp)
        data_dir: Thư mục chứa data
        handle_outliers: Có xử lý outlier weights không (mặc định True)
    
    Returns:
        (coords, dist_matrix)
        coords: list các tuple (x, y) hoặc None nếu EXPLICIT
        dist_matrix: ma trận numpy (N, N)
    """
    if not problem_name.endswith('.tsp'):
        problem_name += '.tsp'
    
    file_path = os.path.join(data_dir, 'tsplib', problem_name)
    if not os.path.exists(file_path):
        file_path = os.path.join(data_dir, 'generated', problem_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy '{problem_name}'")

    metadata = {}
    data_lines = []
    current_section = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'EOF' in line: break
            
            if ':' in line and current_section is None:
                key, value = [s.strip() for s in line.split(':', 1)]
                metadata[key] = value
                continue
            
            # Xử lý GEO không có NODE_COORD_SECTION
            if metadata.get('EDGE_WEIGHT_TYPE') == 'GEO' and 'NODE_COORD_SECTION' not in line and line.replace(" ", "").isdigit():
                 current_section = 'NODE_COORD_SECTION'
                 data_lines.append(line)
                 continue

            if line in ['NODE_COORD_SECTION', 'EDGE_WEIGHT_SECTION', 'DISPLAY_DATA_SECTION']:
                current_section = line
                continue 
            
            if current_section:
                data_lines.append(line)

    try:
        dimension = int(metadata.get('DIMENSION'))
        edge_weight_type = metadata.get('EDGE_WEIGHT_TYPE')
    except Exception as e:
        raise ValueError(f"Lỗi metadata từ {problem_name}: {e}")

    coords = None
    dist_matrix = None
    
    if edge_weight_type in ['EUC_2D', 'GEO', 'ATT']:
        coords = _parse_coords_from_lines(data_lines, dimension)
        
        if edge_weight_type == 'EUC_2D':
            dist_matrix = _calculate_euc_2d_matrix(coords)
        elif edge_weight_type == 'GEO':
            dist_matrix = _calculate_geo_matrix(coords)
        elif edge_weight_type == 'ATT':
            dist_matrix = _calculate_att_matrix(coords)
            
    elif edge_weight_type == 'EXPLICIT':
        coords = None
        edge_weight_format = metadata.get('EDGE_WEIGHT_FORMAT')
        if not edge_weight_format:
            raise ValueError("EXPLICIT thiếu EDGE_WEIGHT_FORMAT")
        dist_matrix = _parse_explicit_matrix(data_lines, dimension, edge_weight_format)
    else:
        raise NotImplementedError(f"EDGE_WEIGHT_TYPE '{edge_weight_type}' chưa hỗ trợ")

    if dist_matrix is None:
        raise ValueError(f"Không thể parse matrix cho {problem_name}")

    # ✅ NEW: Xử lý outlier weights
    if handle_outliers and edge_weight_type == 'EXPLICIT':
        dist_matrix = _handle_outlier_weights(dist_matrix, problem_name)

    return coords, dist_matrix

# --- PHẦN 4: HÀM TẢI OPTIMUM ---

def load_optimum_solution(problem_name, data_dir, dist_matrix):
    """Tải file .opt.tour và tính chi phí tối ưu."""
    if problem_name.endswith('.tsp'):
        problem_name = problem_name.replace('.tsp', '')
            
    file_path = os.path.join(data_dir, 'optimum_solutions', f"{problem_name}.opt.tour")
    
    if not os.path.exists(file_path):
        return None, 0

    tour = []
    in_tour_section = False
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line == 'TOUR_SECTION':
                in_tour_section = True
                continue
            
            if line == '-1' or line == 'EOF': 
                break

            if in_tour_section and line:
                parts = line.split() 
                for part in parts:
                    if part.isdigit():
                        tour.append(int(part) - 1)
                    
    if not tour:
        print(f"LỖI: {problem_name}.opt.tour không đọc được tour")
        return None, 0
    
    if len(tour) != len(dist_matrix):
        print(f"CẢNH BÁO: Tour có {len(tour)} nodes, matrix có {len(dist_matrix)}")
    
    # Chuẩn hóa tour (bắt đầu từ 0)
    if 0 in tour:
        start_index = tour.index(0)
        tour = tour[start_index:] + tour[:start_index]
    else:
        print(f"Cảnh báo: Tour không chứa node 0")
    
    try:
        opt_cost = evaluator.calculate_tour_cost(tour, dist_matrix)
    except Exception as e:
        print(f"LỖI khi tính opt_cost cho {problem_name}: {e}")
        opt_cost = 0

    return tour, opt_cost