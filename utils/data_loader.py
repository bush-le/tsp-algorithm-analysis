import os
import re
import numpy as np
import math  # <- Thư viện mới cần thiết cho GEO và ATT

def _calculate_euc_2d_matrix(coords):
    """
    Tạo ma trận khoảng cách đầy đủ từ tọa độ EUC_2D.
    Sử dụng tính toán vector hóa của NumPy để tăng tốc độ.
    """
    n = len(coords)
    # Chuyển đổi list of tuples [(id, x, y), ...] thành mảng NumPy [[x1, y1], [x2, y2], ...]
    if len(coords[0]) == 3:
        points = np.array([item[1:] for item in coords])
    else:
        points = np.array(coords) # Giả sử là [(x, y), ...]

    # (n, 1, 2) - (1, n, 2) -> (n, n, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    
    # (n, n, 2) -> (n, n)
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Lấy căn bậc hai và làm tròn thành số nguyên (chuẩn nint)
    distances = np.sqrt(dist_sq)
    return np.rint(distances).astype(int)

def _calculate_geo_matrix(coord_lines, dimension):
    """
    Phân tích cú pháp cho NODE_COORD_SECTION (GEO).
    Tính toán ma trận khoảng cách bằng công thức GEO của TSPLIB.
    """
    coords = []
    for line in coord_lines:
        line = line.strip()
        if not line: continue
        try:
            parts = [float(x) for x in line.split()]
            if len(parts) >= 3:
                # Định dạng: (id, lat, lon) - (vĩ độ, kinh độ)
                coords.append((int(parts[0]), parts[1], parts[2]))
        except ValueError:
            continue
            
    if len(coords) != dimension:
        print(f"Cảnh báo: Kích thước (DIMENSION) là {dimension} nhưng tìm thấy {len(coords)} tọa độ.")

    n = dimension
    matrix = np.zeros((n, n), dtype=int)
    R = 6378.388 # Bán kính Trái Đất (km) theo TSPLIB
    
    # Chuyển đổi (vĩ độ, kinh độ) sang radians
    rad_coords = []
    for _, lat_deg, lon_deg in coords:
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
            
            # Công thức khoảng cách GEO của TSPLIB
            # Xử lý lỗi làm tròn số có thể khiến acos(>1.0)
            arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
            arg = min(1.0, max(-1.0, arg)) # kẹp giá trị trong [-1, 1]
                
            dist_ij = R * math.acos(arg) + 1.0
            
            dist_int = int(np.rint(dist_ij)) # Làm tròn số nguyên gần nhất
            
            matrix[i, j] = dist_int
            matrix[j, i] = dist_int
            
    return matrix

def _calculate_att_matrix(coord_lines, dimension):
    """
    Phân tích cú pháp cho NODE_COORD_SECTION (ATT).
    Tính toán ma trận khoảng cách bằng công thức ATT (pseudo-Euclidean) của TSPLIB.
    """
    coords = []
    for line in coord_lines:
        line = line.strip()
        if not line: continue
        try:
            parts = [float(x) for x in line.split()] # Thường là int, nhưng float an toàn hơn
            if len(parts) >= 3:
                # Định dạng: (id, x, y)
                coords.append((int(parts[0]), parts[1], parts[2]))
        except ValueError:
            continue
            
    if len(coords) != dimension:
        print(f"Cảnh báo: Kích thước (DIMENSION) là {dimension} nhưng tìm thấy {len(coords)} tọa độ.")

    n = dimension
    matrix = np.zeros((n, n), dtype=int)
    points = np.array([item[1:] for item in coords])

    for i in range(n):
        for j in range(i + 1, n):
            xd = points[i, 0] - points[j, 0]
            yd = points[i, 1] - points[j, 1]
            
            # Công thức ATT (pseudo-Euclidean)
            r = math.sqrt((xd**2 + yd**2) / 10.0)
            dist_int = int(np.rint(r)) # Làm tròn tới số nguyên gần nhất
            
            matrix[i, j] = dist_int
            matrix[j, i] = dist_int
            
    return matrix

def _parse_coord_data(coord_lines, dimension, coord_type="EUC_2D"):
    """
    Phân tích cú pháp cho NODE_COORD_SECTION và gọi hàm tính toán phù hợp.
    """
    # Trích xuất tọa độ (chung cho cả 3 loại)
    coords = []
    for line in coord_lines:
        line = line.strip()
        if not line:
            continue
        try:
            parts = [float(x) for x in line.split()]
            if len(parts) >= 3:
                coords.append((int(parts[0]), parts[1], parts[2]))
            elif len(parts) == 2:
                coords.append((len(coords) + 1, parts[0], parts[1]))
        except ValueError:
            continue
    
    if len(coords) != dimension:
        print(f"Cảnh báo: Kích thước (DIMENSION) là {dimension} nhưng tìm thấy {len(coords)} tọa độ.")
    
    # Gọi hàm tính toán dựa trên loại
    if coord_type == 'EUC_2D':
        points = [(c[1], c[2]) for c in coords]
        return _calculate_euc_2d_matrix(points)
    
    # (Hai hàm dưới cần danh sách (id, x, y) đầy đủ)
    elif coord_type == 'GEO':
        return _calculate_geo_matrix(coord_lines, dimension)
    
    elif coord_type == 'ATT':
        return _calculate_att_matrix(coord_lines, dimension)
        
    else:
        raise NotImplementedError(f"Loại tọa độ '{coord_type}' chưa được hỗ trợ.")

def _parse_explicit_matrix(matrix_lines, dimension, edge_weight_format):
    """
    Phân tích ma trận khoảng cách từ định dạng EXPLICIT.
    Hỗ trợ: FULL_MATRIX, UPPER_ROW, LOWER_ROW, và LOWER_DIAG_ROW.
    """
    
    data_str = ' '.join(matrix_lines)
    weights = []
    for x in data_str.split():
        try:
            weights.append(int(float(x))) # Chỉ thêm nếu là số hợp lệ
        except ValueError:
            continue # Bỏ qua các chuỗi văn bản

    matrix = np.zeros((dimension, dimension), dtype=int)
    k = 0 # Chỉ số cho list 'weights'

    if edge_weight_format == 'FULL_MATRIX':
        for i in range(dimension):
            for j in range(dimension):
                if k < len(weights):
                    matrix[i, j] = weights[k]
                    k += 1
                else: break
    elif edge_weight_format == 'UPPER_ROW':
        for i in range(dimension):
            for j in range(i + 1, dimension):
                if k < len(weights):
                    matrix[i, j] = weights[k]
                    matrix[j, i] = weights[k] 
                    k += 1
                else: break
    elif edge_weight_format == 'LOWER_ROW':
        for i in range(dimension):
            for j in range(0, i):
                if k < len(weights):
                    matrix[i, j] = weights[k]
                    matrix[j, i] = weights[k] 
                    k += 1
                else: break
    elif edge_weight_format == 'LOWER_DIAG_ROW':
        for i in range(dimension):
            for j in range(i + 1): 
                if k < len(weights):
                    matrix[i, j] = weights[k]
                    matrix[j, i] = weights[k]
                    k += 1
                else: break
    else:
        raise NotImplementedError(f"Định dạng ma trận '{edge_weight_format}' chưa được hỗ trợ.")

    return matrix

def load_problem(problem_name, data_dir):
    """
    Tải một bài toán TSP từ tên file.
    Tự động tìm file .tsp trong /data/tsplib/ hoặc /data/generated/
    và trả về ma trận khoảng cách cùng các thông tin khác.
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

    matrix = None
    
    # === KHỐI LOGIC ĐƯỢC CẬP NHẬT ===
    if edge_weight_type in ['EUC_2D', 'GEO', 'ATT']:
        # Đây là các loại dựa trên NODE_COORD_SECTION
        if 'NODE_COORD_SECTION' not in current_section:
             # Một số file (như GEO) đặt data ngay sau metadata
             pass # Bỏ qua kiểm tra này, _parse_coord_data sẽ xử lý
             
        matrix = _parse_coord_data(data_lines, dimension, coord_type=edge_weight_type)
        
    elif edge_weight_type == 'EXPLICIT':
        edge_weight_format = metadata.get('EDGE_WEIGHT_FORMAT')
        if not edge_weight_format:
            raise ValueError("Lỗi EXPLICIT: Thiếu EDGE_WEIGHT_FORMAT.")
        matrix = _parse_explicit_matrix(data_lines, dimension, edge_weight_format)
    
    else:
        raise NotImplementedError(f"EDGE_WEIGHT_TYPE '{edge_weight_type}' chưa được hỗ trợ.")
    # === KẾT THÚC CẬP NHẬT ===

    if matrix is None:
        raise ValueError(f"Không thể phân tích ma trận cho {problem_name}.")

    return {
        'name': problem_name.replace('.tsp', ''),
        'dimension': dimension,
        'type': edge_weight_type,
        'matrix': matrix
    }

def get_optimum_tour(problem_name, data_dir):
    """
    Tải file .opt.tour (nếu có) từ /data/optimum_solutions/
    Trả về một tour (list các chỉ số) đã được chuẩn hóa (0-indexed).
    """
    if problem_name.endswith('.tsp'):
        problem_name = problem_name.replace('.tsp', '')
        
    file_path = os.path.join(data_dir, 'optimum_solutions', f"{problem_name}.opt.tour")
    
    if not os.path.exists(file_path):
        return None

    tour = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == 'TOUR_SECTION': continue
            if line == '-1' or line == 'EOF': break
            if line.isdigit():
                tour.append(int(line) - 1) # 1-indexed -> 0-indexed
                
    if not tour:
        return None
        
    if 0 in tour:
        start_index = tour.index(0)
        tour = tour[start_index:] + tour[:start_index]
        
    return tour