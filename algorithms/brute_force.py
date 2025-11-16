import numpy as np
import itertools
import sys
import os
from typing import Tuple, List

# --- Import
# Xử lý sys.path để import từ thư mục 'utils'
try:
    from ..utils.evaluator import calculate_tour_cost
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from utils.evaluator import calculate_tour_cost

def solve(matrix: np.ndarray) -> Tuple[List[int], int]:
    """
    Giải TSP bằng thuật toán Vét cạn (Brute Force).

    Thuật toán này kiểm tra tất cả (N-1)! các hoán vị và
    đảm bảo tìm ra giải pháp tối ưu.

    CẢNH BÁO: Độ phức tạp O(N!). Chỉ khả thi cho N rất nhỏ (N <= 10).

    Tuân thủ "Interface" chuẩn: trả về (tour, cost).

    Args:
        matrix (np.ndarray): Ma trận khoảng cách (N x N).

    Returns:
        Tuple[List[int], int]:
            - tour (List[int]): Lộ trình tối ưu (0-indexed).
            - cost (int): Chi phí tối ưu.
    """
    num_cities = matrix.shape[0]
    
    # Trường hợp cơ bản (edge case)
    if num_cities <= 2:
        tour = list(range(num_cities))
        cost = calculate_tour_cost(tour, matrix)
        return tour, int(cost)
        
    min_cost = np.inf
    best_tour = []

    # 1. Cố định nút bắt đầu là 0
    start_node = 0
    
    # 2. Tạo danh sách các nút còn lại để hoán vị
    # (ví dụ: [1, 2, 3, ..., N-1])
    other_nodes = list(range(1, num_cities))

    # 3. Lặp qua tất cả (N-1)! hoán vị
    for perm in itertools.permutations(other_nodes):
        # 'perm' là một tuple, ví dụ: (2, 1, 3)
        
        # 4. Tạo lộ trình đầy đủ
        # Ví dụ: [0] + [2, 1, 3] = [0, 2, 1, 3]
        current_tour = [start_node] + list(perm)
        
        # 5. Tính chi phí (evaluator của chúng ta sẽ tự động
        #    thêm chi phí quay lại nút 0)
        current_cost = calculate_tour_cost(current_tour, matrix)
        
        # 6. Cập nhật nếu tốt hơn
        if current_cost < min_cost:
            min_cost = current_cost
            best_tour = current_tour
            
    return best_tour, int(min_cost)