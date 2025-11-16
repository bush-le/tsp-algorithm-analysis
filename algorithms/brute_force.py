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

# --- Ví dụ sử dụng (chỉ để kiểm tra nhanh) ---
if __name__ == "__main__":
    # 1. Sử dụng ma trận 5x5
    test_matrix = np.array([
        [0, 3, 4, 5, 1], # 0
        [3, 0, 5, 1, 6], # 1
        [4, 5, 0, 2, 7], # 2
        [5, 1, 2, 0, 3], # 3
        [1, 6, 7, 3, 0]  # 4
    ])
    
    print("--- Chạy kiểm tra Brute Force (N=5) ---")
    print(f"Số hoán vị cần kiểm tra: {(5-1)!} = 24")
    
    tour, cost = solve(test_matrix)
    
    print(f"Tour tối ưu: {tour}")
    print(f"Chi phí tối ưu: {cost}")
    
    # Chúng ta biết chi phí tối ưu là 14 từ các lần chạy trước
    assert cost == 14
    print("Kiểm tra (N=5) thành công!")

    # 2. Sử dụng ma trận 4x4 (từ Wikipedia)
    test_matrix_4 = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    
    print("\n--- Chạy kiểm tra Brute Force (N=4) ---")
    print(f"Số hoán vị cần kiểm tra: {(4-1)!} = 6")
    
    tour_4, cost_4 = solve(test_matrix_4)
    
    # Kết quả mong đợi: [0, 1, 3, 2]
    # Chi phí: M[0,1]+M[1,3]+M[3,2]+M[2,0] = 10 + 25 + 30 + 15 = 80
    print(f"Tour tối ưu: {tour_4}")
    print(f"Chi phí tối ưu: {cost_4}")
    
    assert cost_4 == 80
    print("Kiểm tra (N=4) thành công!")