import os
import re
import numpy as np
from .distance import calculate_distance_matrix

def _load_optimum_solution(problem_name: str, data_dir: str = "data") -> list[int] | None:
    """
    Tải file giải pháp tối ưu (.opt.tour) nếu tồn tại.
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

def load_problem(problem_name: str, data_dir: str = "data") -> dict:
    """
    Tải và phân tích cú pháp (parse) một file vấn đề TSPLIB (.tsp).
    
    Hàm này sẽ tìm file trong các thư mục con đã biết (tsplib, generated).
    """
    
    # --- ĐÂY LÀ THAY ĐỔI QUAN TRỌNG ---
    search_dirs = ["tsplib", "generated"]
    tsp_path = None
    
    for subdir in search_dirs:
        path = os.path.join(data_dir, subdir, f"{problem_name}.tsp")
        if os.path.exists(path):
            tsp_path = path
            break # Dừng ngay khi tìm thấy file
            
    if tsp_path is None:
        raise FileNotFoundError(f"Không tìm thấy file {problem_name}.tsp trong "
                                f"{[os.path.join(data_dir, d) for d in search_dirs]}")
    # --- KẾT THÚC THAY ĐỔI ---

    coords = []
    dimension = 0
    reading_coords = False
    problem_name_from_file = problem_name

    with open(tsp_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if ":" in line:
                key, value = [s.strip() for s in line.split(":", 1)] 
                if key == "NAME":
                    problem_name_from_file = value
                elif key == "DIMENSION":
                    try:
                        dimension = int(value)
                    except ValueError:
                        raise ValueError(f"Giá trị DIMENSION không hợp lệ: {value}")
                # Sau khi xử lý dòng metadata, chuyển sang dòng tiếp theo
                continue
            
            if line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            elif line.startswith("EOF"):
                break
                
            if reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        coords.append([float(parts[1]), float(parts[2])])
                    except ValueError:
                        print(f"Cảnh báo: Bỏ qua dòng không hợp lệ: {line}")

    if not coords or len(coords) != dimension:
        raise ValueError(f"Lỗi phân tích {problem_name_from_file}: "
                         f"DIMENSION ({dimension}) không khớp số tọa độ ({len(coords)}).")

    coords_np = np.array(coords)
    dist_matrix = calculate_distance_matrix(coords_np)
    opt_tour = _load_optimum_solution(problem_name, data_dir)
    
    return {
        "name": problem_name_from_file,
        "dimension": dimension,
        "coords": coords_np,
        "matrix": dist_matrix,
        "optimum_tour": opt_tour
    }