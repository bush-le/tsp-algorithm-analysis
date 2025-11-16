import os
import re
import numpy as np
import math
from . import evaluator # Chúng ta cần evaluator để tính opt_cost

# --- PHẦN 1: CÁC HÀM TÍNH TOÁN KHOẢNG CÁCH ---
# (Phần này của bạn đã chính xác, giữ nguyên)

def _calculate_euc_2d_matrix(coords):
    """
    Tạo ma trận khoảng cách EUC_2D từ list tọa độ (x, y).
    Sử dụng NumPy vector hóa.
    """
    points = np.array(coords) # Yêu cầu đầu vào là list [[x1, y1], [x2, y2], ...]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    distances = np.sqrt(dist_sq)
    return np.rint(distances).astype(int)

def _calculate_geo_matrix(coords):
    """
    Tạo ma trận khoảng cách GEO từ list tọa độ (lat, lon).
    """
    n = len(coords)
    matrix = np.zeros((n, n), dtype=int)
    R = 6378.388 # Bán kính Trái Đất (km) theo TSPLIB
    
    # Chuyển đổi (vĩ độ, kinh độ) sang radians
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
            arg = min(1.0, max(-1.0, arg)) # Kẹp giá trị trong [-1, 1]
            
            dist_ij = R * math.acos(arg) + 1.0
            dist_int = int(np.rint(dist_ij))
            
            matrix[i, j] = dist_int
            matrix[j, i] = dist_int
            
    return matrix

def _calculate_att_matrix(coords):
    """
    Tạo ma trận khoảng cách ATT (pseudo-Euclidean) từ list tọa độ (x, y).
    """
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
# (Phần này của bạn đã chính xác, giữ nguyên)

def _parse_coords_from_lines(coord_lines, dimension):
    """
    Hàm phân tích NODE_COORD_SECTION (hoặc tương đương).
    Trả về: list các tọa độ [[x1, y1], [x2, y2], ...]
    """
    coords = []
    for line in coord_lines:
        line = line.strip()
        if not line: continue
        try:
            parts = [float(x) for x in line.split()]
            if len(parts) >= 3:
                # Định dạng: (id, x, y) hoặc (id, lat, lon)
                coords.append((int(parts[0]), parts[1], parts[2]))
            elif len(parts) == 2:
                # Định dạng: (x, y) (thường thấy trong ATT)
                coords.append((len(coords) + 1, parts[0], parts[1]))
        except ValueError:
            continue
            
    if len(coords) != dimension:
        print(f"Cảnh báo: Kích thước (DIMENSION) là {dimension} nhưng tìm thấy {len(coords)} tọa độ.")
    
    # Sắp xếp lại theo ID (quan trọng) và chỉ lấy (x, y)
    # TSPLIB là 1-indexed
    coords.sort(key=lambda x: x[0]) 
    final_coords = [(c[1], c[2]) for c in coords]
    
    return final_coords

def _parse_explicit_matrix(matrix_lines, dimension, edge_weight_format):
    """
    Phân tích ma trận khoảng cách từ định dạng EXPLICIT (ví dụ: gr24).
    """
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
    elif edge_weight_format == 'LOWER_DIAG_ROW': # Đây là loại của gr24
        for i in range(dimension):
            for j in range(i + 1): 
                if k < len(weights):
                    matrix[i, j] = weights[k]
                    matrix[j, i] = weights[k]
                    k += 1
    else:
        raise NotImplementedError(f"Định dạng ma trận '{edge_weight_format}' chưa được hỗ trợ.")

    return matrix

# --- PHẦN 3: HÀM GIAO DIỆN CHÍNH (PUBLIC) ---
# (Hàm load_tsp_problem của bạn đã chính xác, giữ nguyên)

def load_tsp_problem(problem_name, data_dir):
    """
    Hàm tải chính, thay thế tất cả các hàm tải cũ.
    Tải một bài toán TSP từ tên file, hỗ trợ EUC_2D, GEO, ATT, và EXPLICIT.
    
    Trả về:
        (coords, dist_matrix)
        
    'coords' là list các tuple (x, y) hoặc (lat, lon)
    'coords' sẽ là None nếu loại là EXPLICIT.
    'dist_matrix' là ma trận NumPy (N, N).
    """
    if not problem_name.endswith('.tsp'):
        problem_name += '.tsp'
    
    file_path = os.path.join(data_dir, 'tsplib', problem_name)
    if not os.path.exists(file_path):
        file_path = os.path.join(data_dir, 'generated', problem_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file '{problem_name}' trong cả 'tsplib' và 'generated'")

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
            
            # Xử lý trường hợp file GEO không có 'NODE_COORD_SECTION'
            if metadata.get('EDGE_WEIGHT_TYPE') == 'GEO' and 'NODE_COORD_SECTION' not in line and line.replace(" ", "").isdigit():
                 current_section = 'NODE_COORD_SECTION' # Giả lập
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
        raise ValueError(f"Lỗi khi đọc metadata (DIMENSION, EDGE_WEIGHT_TYPE) từ {problem_name}: {e}")

    coords = None
    dist_matrix = None
    
    if edge_weight_type in ['EUC_2D', 'GEO', 'ATT']:
        # 1. Phân tích tọa độ
        coords = _parse_coords_from_lines(data_lines, dimension)
        
        # 2. Tính toán ma trận
        if edge_weight_type == 'EUC_2D':
            dist_matrix = _calculate_euc_2d_matrix(coords)
        elif edge_weight_type == 'GEO':
            dist_matrix = _calculate_geo_matrix(coords)
        elif edge_weight_type == 'ATT':
            dist_matrix = _calculate_att_matrix(coords)
            
    elif edge_weight_type == 'EXPLICIT':
        # 1. Tọa độ là None
        coords = None
        
        # 2. Phân tích ma trận
        edge_weight_format = metadata.get('EDGE_WEIGHT_FORMAT')
        if not edge_weight_format:
            raise ValueError("Lỗi EXPLICIT: Thiếu EDGE_WEIGHT_FORMAT.")
        dist_matrix = _parse_explicit_matrix(data_lines, dimension, edge_weight_format)
    
    else:
        raise NotImplementedError(f"EDGE_WEIGHT_TYPE '{edge_weight_type}' chưa được hỗ trợ.")

    if dist_matrix is None:
        raise ValueError(f"Không thể phân tích ma trận cho {problem_name}.")

    # Trả về theo giao diện chuẩn (coords, matrix)
    return coords, dist_matrix

# --- PHẦN 4: HÀM TẢI OPTIMUM (ĐÃ SỬA LỖI) ---
# *** ĐÂY LÀ PHẦN ĐƯỢC THAY THẾ ***

def load_optimum_solution(problem_name, data_dir, dist_matrix):
    """
    Tải file .opt.tour VÀ tính chi phí tối ưu.
    *** ĐÃ NÂNG CẤP ĐỂ ĐỌC ĐỊNH DẠNG TOUR_SECTION DẠNG KHỐI (BLOCK) ***
    """
    if problem_name.endswith('.tsp'):
        problem_name = problem_name.replace('.tsp', '')
            
    file_path = os.path.join(data_dir, 'optimum_solutions', f"{problem_name}.opt.tour")
    
    if not os.path.exists(file_path):
        return None, 0 # Không có tour tối ưu

    tour = []
    in_tour_section = False # Cờ để biết khi nào bắt đầu đọc
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line == 'TOUR_SECTION':
                in_tour_section = True
                continue
            
            if line == '-1' or line == 'EOF': 
                break # Dừng khi gặp -1 hoặc EOF

            if in_tour_section and line: # Chỉ xử lý nếu đang trong section và dòng không rỗng
                # --- ĐÂY LÀ PHẦN SỬA LỖI QUAN TRỌNG ---
                # Tách dòng thành các phần (ví dụ: "29 7 28...")
                parts = line.split() 
                for part in parts:
                    if part.isdigit():
                        tour.append(int(part) - 1) # 1-indexed -> 0-indexed
                # --- KẾT THÚC SỬA LỖI ---
                    
    if not tour:
        print(f"LỖI: {problem_name}.opt.tour được tìm thấy nhưng không thể đọc tour.")
        return None, 0
    
    # Kiểm tra xem có đọc đủ thành phố không
    if len(tour) != len(dist_matrix):
        print(f"CẢNH BÁO: Đọc {len(tour)} thành phố từ .opt.tour, nhưng bài toán có {len(dist_matrix)} thành phố.")
        # Có thể file tour bị lỗi, nhưng chúng ta vẫn thử tính
    
    # Chuẩn hóa tour (bắt đầu từ 0)
    if 0 in tour:
        start_index = tour.index(0)
        tour = tour[start_index:] + tour[:start_index]
    else:
        print(f"Cảnh báo: Tour của {problem_name} không chứa nút 0. Sử dụng tour gốc.")

    
    # Tính toán chi phí
    try:
        opt_cost = evaluator.calculate_tour_cost(tour, dist_matrix)
    except Exception as e:
        print(f"LỖI khi tính opt_cost cho {problem_name}: {e}")
        opt_cost = 0

    return tour, opt_cost