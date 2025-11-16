import os
import re
import numpy as np
from .distance import calculate_distance_matrix

def _load_optimum_solution(problem_name: str, data_dir: str = "data") -> list[int] | None:
    """
    Tải file giải pháp tối ưu (.opt.tour) nếu tồn tại.

    Các file này chứa một danh sách các ID nút (1-indexed).
    Hàm này sẽ chuyển đổi chúng thành 0-indexed.
    """
    path = os.path.join(data_dir, "optimum_solutions", f"{problem_name}.opt.tour")
    
    if not os.path.exists(path):
        print(f"Lưu ý: Không tìm thấy file giải pháp tối ưu cho {problem_name}.")
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
                    # Chuyển đổi ID nút từ 1-indexed (TSPLIB) sang 0-indexed (Python)
                    tour.append(int(line) - 1)
        
        return tour if tour else None
        
    except Exception as e:
        print(f"Lỗi khi đọc file tối ưu {path}: {e}")
        return None

def load_problem(problem_name: str, data_dir: str = "data") -> dict:
    """
    Tải và phân tích cú pháp (parse) một file vấn đề TSPLIB (.tsp).

    Hàm này đọc file, trích xuất siêu dữ liệu (tên, kích thước) và
    quan trọng nhất là tọa độ (NODE_COORD_SECTION).

    Sau đó, nó sử dụng calculate_distance_matrix để tạo ma trận.

    Args:
        problem_name (str): Tên của vấn đề (ví dụ: 'berlin52').
        data_dir (str): Thư mục gốc chứa '/tsplib/' và '/optimum_solutions/'.

    Returns:
        dict: Một dictionary chứa:
            - 'name': Tên vấn đề (str)
            - 'dimension': Số lượng thành phố (int)
            - 'coords': Mảng (N, 2) của tọa độ (np.ndarray)
            - 'matrix': Ma trận khoảng cách (N, N) (np.ndarray)
            - 'optimum_tour': Lộ trình tối ưu (list[int], 0-indexed) hoặc None
    """
    tsp_path = os.path.join(data_dir, "tsplib", f"{problem_name}.tsp")
    
    if not os.path.exists(tsp_path):
        raise FileNotFoundError(f"Không tìm thấy file vấn đề: {tsp_path}")

    coords = []
    dimension = 0
    reading_coords = False

    with open(tsp_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("NAME"):
                # Tên có thể là "NAME: berlin52"
                problem_name = line.split(":")[-1].strip()
            elif line.startswith("DIMENSION"):
                # Kích thước có thể là "DIMENSION : 52"
                dimension = int(line.split(":")[-1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            elif line.startswith("EOF"):
                break
                
            if reading_coords:
                # Dòng dữ liệu: "1 565.0 575.0"
                parts = line.split()
                if len(parts) >= 3:
                    # Bỏ qua ID nút (phần tử đầu tiên), chỉ lấy x, y
                    coords.append([float(parts[1]), float(parts[2])])

    if not coords or len(coords) != dimension:
        raise ValueError(f"Lỗi khi phân tích cú pháp file {problem_name}. Kích thước không khớp.")

    # Chuyển đổi tọa độ sang NumPy array
    coords_np = np.array(coords)
    
    # Sử dụng module 'distance' của chúng ta!
    dist_matrix = calculate_distance_matrix(coords_np)
    
    # Tải giải pháp tối ưu (nếu có)
    opt_tour = _load_optimum_solution(problem_name, data_dir)
    
    return {
        "name": problem_name,
        "dimension": dimension,
        "coords": coords_np,
        "matrix": dist_matrix,
        "optimum_tour": opt_tour
    }

# --- Ví dụ sử dụng (chỉ để kiểm tra nhanh) ---
if __name__ == "__main__":
    # Để chạy kiểm tra này, hãy đảm bảo bạn có file:
    # ./data/tsplib/berlin52.tsp
    # ./data/optimum_solutions/berlin52.opt.tour
    
    try:
        # Giả sử chúng ta đang chạy file này từ thư mục gốc của dự án
        # và cấu trúc thư mục là /data/tsplib/berlin52.tsp
        # Nếu chạy trực tiếp từ /utils/, chúng ta cần điều chỉnh đường dẫn
        # Giả định chạy từ gốc (hoặc 'data' nằm cùng cấp với 'utils')
        
        # Đường dẫn tương đối từ 'utils' lên thư mục gốc rồi xuống 'data'
        project_root_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        # Chuẩn hóa đường dẫn
        project_root_data_dir = os.path.normpath(project_root_data_dir)

        # Kiểm tra xem 'data' có tồn tại ở đường dẫn dự kiến không
        if not os.path.isdir(project_root_data_dir):
            print("Không tìm thấy thư mục 'data' ở cấp gốc. Thử ở thư mục hiện tại.")
            project_root_data_dir = "data" # Mặc định nếu chạy từ gốc
            
        print(f"Sử dụng thư mục data: {project_root_data_dir}")

        problem_name = "berlin52"
        problem_data = load_problem(problem_name, data_dir=project_root_data_dir)
        
        print(f"\n--- Kiểm tra {problem_name} ---")
        print(f"Tên: {problem_data['name']}")
        print(f"Kích thước: {problem_data['dimension']}")
        print(f"Hình dạng tọa độ: {problem_data['coords'].shape}")
        print(f"Hình dạng ma trận: {problem_data['matrix'].shape}")
        
        if problem_data['optimum_tour']:
            print(f"Đã tải lộ trình tối ưu (0-indexed, 5 nút đầu): {problem_data['optimum_tour'][:5]}...")
            print(f"Tổng số nút trong tour tối ưu: {len(problem_data['optimum_tour'])}")
            assert len(problem_data['optimum_tour']) == problem_data['dimension']
        else:
            print("Không tìm thấy lộ trình tối ưu cho berlin52 (kiểm tra đường dẫn).")

        assert problem_data['dimension'] == 52
        assert problem_data['coords'].shape == (52, 2)
        assert problem_data['matrix'].shape == (52, 52)
        
        print("\nKiểm tra Data Loader thành công!")

    except FileNotFoundError as e:
        print(f"\nLỗi: Không tìm thấy file. {e}")
        print("Hãy đảm bảo bạn đã tải file 'berlin5s2.tsp' và 'berlin52.opt.tour'")
        print("vào đúng các thư mục /data/tsplib/ và /data/optimum_solutions/")
    except Exception as e:
        print(f"\nLỗi khi chạy kiểm tra: {e}")