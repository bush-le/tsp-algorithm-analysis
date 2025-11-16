import os
import re
import numpy as np
import math

def _calculate_euc_2d_matrix(coords):
    """
    Tạo ma trận khoảng cách đầy đủ từ tọa độ EUC_2D.
    Sử dụng tính toán vector hóa của NumPy để tăng tốc độ.
    """
    n = len(coords)
    # Chuyển đổi list of tuples [(id, x, y), ...] thành mảng NumPy [[x1, y1], [x2, y2], ...]
    # Giả định coords đã được lọc và chỉ chứa (x, y) hoặc (id, x, y)
    if len(coords[0]) == 3:
        points = np.array([item[1:] for item in coords])
    else:
        points = np.array(coords)

    # Tính toán sự khác biệt bình phương
    # (n, 1, 2) - (1, n, 2) -> (n, n, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    
    # (n, n, 2) -> (n, n)
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Lấy căn bậc hai và làm tròn thành số nguyên
    distances = np.sqrt(dist_sq)
    return np.rint(distances).astype(int)

def _parse_coord_data(coord_lines, dimension):
    """
    Phân tích cú pháp cho NODE_COORD_SECTION.
    """
    coords = []
    for line in coord_lines:
        line = line.strip()
        if not line:
            continue
        try:
            # Tách dữ liệu, hỗ trợ cả "1 10.5 20.3" và "1 10 20"
            parts = [float(x) for x in line.split()]
            if len(parts) >= 3:
                # Định dạng: (id, x, y)
                coords.append((int(parts[0]), parts[1], parts[2]))
            elif len(parts) == 2:
                 # Định dạng: (x, y) - một số file không có id
                coords.append((len(coords) + 1, parts[0], parts[1]))
        except ValueError:
            # Bỏ qua các dòng không hợp lệ
            continue
    
    if len(coords) != dimension:
        print(f"Cảnh báo: Kích thước (DIMENSION) là {dimension} nhưng tìm thấy {len(coords)} tọa độ.")
        
    # Chỉ lấy các tọa độ (x, y) để tính toán
    points = [(c[1], c[2]) for c in coords]
    return _calculate_euc_2d_matrix(points)

def _parse_explicit_matrix(matrix_lines, dimension, edge_weight_format):
    """
    Phân tích ma trận khoảng cách từ định dạng EXPLICIT.
    Hỗ trợ: FULL_MATRIX, UPPER_ROW, LOWER_ROW, và LOWER_DIAG_ROW.
    """
    
    data_str = ' '.join(matrix_lines)
    
    # === BẢN VÁ LỖI (HOTFIX) ===
    # Sử dụng try-except để lọc các giá trị không phải số
    # như 'DISPLAY_DATA_SECTION', 'EOF', v.v.
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
                else:
                    print(f"Lỗi: Dữ liệu FULL_MATRIX không đủ. Dừng ở ({i}, {j})")
                    return matrix

    elif edge_weight_format == 'UPPER_ROW':
        for i in range(dimension):
            for j in range(i + 1, dimension):
                if k < len(weights):
                    matrix[i, j] = weights[k]
                    matrix[j, i] = weights[k] # Đối xứng
                    k += 1
                else:
                    print(f"Lỗi: Dữ liệu UPPER_ROW không đủ. Dừng ở ({i}, {j})")
                    return matrix

    elif edge_weight_format == 'LOWER_ROW':
        for i in range(dimension):
            for j in range(0, i):
                if k < len(weights):
                    matrix[i, j] = weights[k]
                    matrix[j, i] = weights[k] # Đối xứng
                    k += 1
                else:
                    print(f"Lỗi: Dữ liệu LOWER_ROW không đủ. Dừng ở ({i}, {j})")
                    return matrix
    
    # === KHỐI CODE MỚI ĐƯỢC THÊM VÀO ===
    elif edge_weight_format == 'LOWER_DIAG_ROW':
        for i in range(dimension):
            # j chạy từ 0 đến i (bao gồm cả i)
            for j in range(i + 1): 
                if k < len(weights):
                    matrix[i, j] = weights[k]
                    matrix[j, i] = weights[k] # Đối xứng
                    k += 1
                else:
                    print(f"Lỗi: Dữ liệu LOWER_DIAG_ROW không đủ. Dừng ở ({i}, {j})")
                    return matrix
    # === KẾT THÚC KHỐI CODE MỚI ===

    else:
        raise NotImplementedError(f"Định dạng ma trận '{edge_weight_format}' chưa được hỗ trợ.")

    return matrix

def load_problem(problem_name, data_dir):
    """
    Tải một bài toán TSP từ tên file.
    Tự động tìm file .tsp trong /data/tsplib/ hoặc /data/generated/
    và trả về ma trận khoảng cách cùng các thông tin khác.
    """
    # Xây dựng đường dẫn file
    if not problem_name.endswith('.tsp'):
        problem_name += '.tsp'
    
    # Ưu tiên tìm trong tsplib trước
    file_path = os.path.join(data_dir, 'tsplib', problem_name)
    if not os.path.exists(file_path):
        # Nếu không thấy, tìm trong generated
        file_path = os.path.join(data_dir, 'generated', problem_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file '{problem_name}' trong cả 'tsplib' và 'generated'")

    # Đọc metadata và data
    metadata = {}
    data_lines = []
    current_section = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if 'EOF' in line:
                break
            
            # 1. Đọc Metadata (trước khi vào data section)
            if ':' in line and current_section is None:
                key, value = [s.strip() for s in line.split(':', 1)]
                metadata[key] = value
                continue

            # 2. Xác định Data Section
            if line in ['NODE_COORD_SECTION', 'EDGE_WEIGHT_SECTION', 'DISPLAY_DATA_SECTION']:
                current_section = line
                continue # Bỏ qua chính dòng tiêu đề
            
            # 3. Đọc Data
            if current_section:
                data_lines.append(line)

    # Lấy metadata quan trọng
    try:
        dimension = int(metadata.get('DIMENSION'))
        edge_weight_type = metadata.get('EDGE_WEIGHT_TYPE')
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc metadata (DIMENSION, EDGE_WEIGHT_TYPE) từ {problem_name}: {e}")

    # Xử lý dữ liệu dựa trên loại
    matrix = None
    if edge_weight_type == 'EUC_2D':
        matrix = _parse_coord_data(data_lines, dimension)
        
    elif edge_weight_type == 'EXPLICIT':
        edge_weight_format = metadata.get('EDGE_WEIGHT_FORMAT')
        if not edge_weight_format:
            raise ValueError("Lỗi EXPLICIT: Thiếu EDGE_WEIGHT_FORMAT.")
        matrix = _parse_explicit_matrix(data_lines, dimension, edge_weight_format)
    
    elif edge_weight_type == 'ATT':
        # ATT (pseudo-Euclidean) là một trường hợp đặc biệt, phức tạp hơn
        # Tạm thời, chúng ta sẽ coi nó giống EUC_2D nhưng dùng hàm tính khoảng cách khác
        print(f"Cảnh báo: Loại 'ATT' được xử lý như 'EUC_2D'. Cần kiểm tra lại công thức tính khoảng cách nếu kết quả sai.")
        matrix = _parse_coord_data(data_lines, dimension)
        
    else:
        raise NotImplementedError(f"EDGE_WEIGHT_TYPE '{edge_weight_type}' chưa được hỗ trợ.")

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
        # print(f"Lưu ý: Không tìm thấy file optimum tour cho {problem_name}.")
        return None

    tour = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == 'TOUR_SECTION':
                continue
            if line == '-1' or line == 'EOF':
                break
            if line.isdigit():
                # TSPLIB dùng 1-indexed, chúng ta chuyển về 0-indexed
                tour.append(int(line) - 1)
                
    if not tour:
        return None
        
    # Đảm bảo tour bắt đầu từ 0
    if 0 in tour:
        start_index = tour.index(0)
        tour = tour[start_index:] + tour[:start_index]
        
    return tour