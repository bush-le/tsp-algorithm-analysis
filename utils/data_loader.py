import os
import re
import numpy as np
# Import hàm tính khoảng cách (vectorized) đúng từ file distance.py
from .distance import calculate_distance_matrix 

def _load_optimum_solution(problem_name: str, data_dir: str = "data") -> list[int] | None:
    """
    Tải file giải pháp tối ưu (.opt.tour) nếu tồn tại.
    Chuyển đổi từ 1-indexed (TSPLIB) sang 0-indexed (Python).
    """
    path = os.path.join(data_dir, "optimum_solutions", f"{problem_name}.opt.tour")
    
    if not os.path.exists(path):
        return None
        
    tour = []
    reading_tour = False
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("TOUR_SECTION"):
                    reading_tour = True
                    continue
                if line == "-1" or line.startswith("EOF"):
                    break
                if reading_tour:
                    try:
                        node_id = int(line)
                        if node_id > 0:
                            tour.append(node_id - 1) 
                    except ValueError:
                        continue 
        
        return tour if tour else None
        
    except Exception as e:
        print(f"Lỗi khi đọc file tối ưu {path}: {e}")
        return None

def _parse_explicit_matrix(matrix_lines, dimension, edge_weight_format):
    """
    Phân tích ma trận khoảng cách từ định dạng EXPLICIT.
    Hỗ trợ: FULL_MATRIX và UPPER_ROW.
    """
    
    # Ghép tất cả các dòng dữ liệu thành một chuỗi và tách thành các số
    data_str = ' '.join(matrix_lines)
    weights = [int(float(x)) for x in data_str.split()] # Làm tròn thành int
    
    matrix = np.zeros((dimension, dimension), dtype=int)
    k = 0 # Con trỏ cho mảng weights

    if edge_weight_format == 'FULL_MATRIX':
        for i in range(dimension):
            for j in range(dimension):
                if k >= len(weights): raise ValueError("Không đủ dữ liệu cho FULL_MATRIX.")
                matrix[i, j] = weights[k]
                k += 1
                
    elif edge_weight_format == 'UPPER_ROW':
        for i in range(dimension):
            for j in range(i + 1, dimension):
                if k >= len(weights): raise ValueError("Không đủ dữ liệu cho UPPER_ROW.")
                matrix[i, j] = weights[k]
                matrix[j, i] = weights[k] # Ma trận đối xứng
                k += 1
    else:
        raise NotImplementedError(f"Định dạng ma trận '{edge_weight_format}' chưa được hỗ trợ.")
        
    return matrix

def load_problem(problem_name: str, data_dir: str = "data") -> dict:
    """
    Tải và phân tích cú pháp (parse) một file vấn đề TSPLIB (.tsp).
    Hỗ trợ EUC_2D và EXPLICIT.
    Tìm kiếm trong 'tsplib' và 'generated'.
    """
    
    # 1. LOGIC TÌM KIẾM (Đã thống nhất)
    search_dirs = ["tsplib", "generated"]
    tsp_path = None
    
    for subdir in search_dirs:
        path = os.path.join(data_dir, subdir, f"{problem_name}.tsp")
        if os.path.exists(path):
            tsp_path = path
            break
            
    if tsp_path is None:
        raise FileNotFoundError(f"Không tìm thấy file {problem_name}.tsp trong "
                                f"{[os.path.join(data_dir, d) for d in search_dirs]}")

    # 2. Phân tích file .tsp
    coords = []
    dimension = 0
    reading_coords = False
    reading_matrix = False
    matrix_lines = []
    
    problem_name_from_file = problem_name
    edge_weight_type = ""
    edge_weight_format = ""

    with open(tsp_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if ":" in line:
                key, value = [s.strip() for s in line.split(":", 1)] 
                if key == "NAME": problem_name_from_file = value
                elif key == "DIMENSION": dimension = int(value)
                elif key == "EDGE_WEIGHT_TYPE": edge_weight_type = value
                elif key == "EDGE_WEIGHT_FORMAT": edge_weight_format = value
                continue
            
            if line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            if line.startswith("EDGE_WEIGHT_SECTION"):
                reading_matrix = True
                continue
            elif line.startswith("EOF"):
                break
                
            if reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        coords.append([float(parts[1]), float(parts[2])])
                    except ValueError:
                        print(f"Cảnh báo: Bỏ qua dòng tọa độ: {line}")
                        
            if reading_matrix:
                matrix_lines.append(line)

    # 3. Tạo Ma trận
    dist_matrix = None
    coords_np = None

    if edge_weight_type == "EUC_2D":
        if not coords or len(coords) != dimension:
            raise ValueError(f"Lỗi EUC_2D: Kích thước {dimension} không khớp {len(coords)} tọa độ.")
        coords_np = np.array(coords)
        # GỌI HÀM TÍNH KHOẢNG CÁCH ĐÚNG (TỪ UTILS/DISTANCE.PY)
        dist_matrix = calculate_distance_matrix(coords_np)
        
    elif edge_weight_type == "EXPLICIT":
        if not edge_weight_format:
            raise ValueError("Lỗi EXPLICIT: Thiếu EDGE_WEIGHT_FORMAT.")
        dist_matrix = _parse_explicit_matrix(matrix_lines, dimension, edge_weight_format)
        
    else:
        raise NotImplementedError(f"EDGE_WEIGHT_TYPE '{edge_weight_type}' chưa được hỗ trợ.")

    # 4. TẢI GIẢI PHÁP TỐI ƯU (CỰC KỲ QUAN TRỌNG)
    opt_tour = _load_optimum_solution(problem_name, data_dir)
    
    return {
        "name": problem_name_from_file,
        "dimension": dimension,
        "coords": coords_np, # Sẽ là None nếu là EXPLICIT
        "matrix": dist_matrix,
        "optimum_tour": opt_tour # <-- Trả về tour tối ưu
    }