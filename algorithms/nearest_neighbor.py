import numpy as np
import sys
import os
from typing import Tuple, List

# --- Import
# Thêm đường dẫn gốc của dự án vào sys.path để cho phép import
# từ thư mục 'utils' (ví dụ: from ..utils.evaluator import ...)
# Điều này cần thiết để tuân thủ cấu trúc module của chúng ta
try:
    from ..utils.evaluator import calculate_tour_cost
except ImportError:
    # Đoạn này xử lý trường hợp khi chạy file trực tiếp (cho __main__)
    # Nó thêm thư mục cha (gốc dự án) vào sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from utils.evaluator import calculate_tour_cost

def solve(matrix: np.ndarray, start_node: int = 0) -> Tuple[List[int], int]:
    """
    Giải TSP bằng thuật toán Láng giềng gần nhất (Nearest Neighbor).

    Bắt đầu từ 'start_node' và lặp đi lặp lại việc di chuyển đến
    thành phố chưa được thăm gần nhất.

    Tuân thủ "Interface" chuẩn: trả về (tour, cost).

    Args:
        matrix (np.ndarray): Ma trận khoảng cách (N x N).
        start_node (int): Nút bắt đầu (0-indexed).

    Returns:
        Tuple[List[int], int]:
            - tour (List[int]): Danh sách các ID nút (0-indexed).
            - cost (int): Tổng chi phí của lộ trình.
    """
    num_cities = matrix.shape[0]
    if num_cities == 0:
        return [], 0
        
    tour = [start_node]
    visited = np.zeros(num_cities, dtype=bool)
    visited[start_node] = True
    
    current_node = start_node
    
    # Chúng ta cần chọn N-1 nút tiếp theo
    for _ in range(num_cities - 1):
        # Lấy hàng khoảng cách từ nút hiện tại và đảm bảo nó là kiểu float
        # để có thể gán giá trị np.inf
        distances = matrix[current_node].copy().astype(float)
        
        # Đặt khoảng cách đến các nút đã thăm là vô cùng
        # để đảm bảo chúng không được chọn
        distances[visited] = np.inf
        
        # Tìm nút chưa thăm gần nhất
        next_node = np.argmin(distances)
        
        # Kiểm tra nếu không tìm thấy (ví dụ: ma trận toàn inf)
        if distances[next_node] == np.inf:
            break # Bị mắc kẹt, dừng sớm (không nên xảy ra trong TSP)
            
        # Thêm nút vào tour và đánh dấu đã thăm
        tour.append(next_node)
        visited[next_node] = True
        current_node = next_node
        
    # Tính chi phí cuối cùng (bao gồm cả cạnh quay về)
    # bằng cách sử dụng evaluator của chúng ta
    cost = calculate_tour_cost(tour, matrix)
    
    return tour, int(cost)
