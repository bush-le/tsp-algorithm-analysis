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
        # Lấy hàng khoảng cách từ nút hiện tại
        distances = matrix[current_node].copy()
        
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

# --- Ví dụ sử dụng (chỉ để kiểm tra nhanh) ---
if __name__ == "__main__":
    # Sử dụng lại ma trận 5x5 từ ví dụ của evaluator
    test_matrix = np.array([
        [0, 3, 4, 5, 1], # 0
        [3, 0, 5, 1, 6], # 1
        [4, 5, 0, 2, 7], # 2
        [5, 1, 2, 0, 3], # 3
        [1, 6, 7, 3, 0]  # 4
    ])
    
    print("--- Chạy kiểm tra Nearest Neighbor ---")
    print("Ma trận thử nghiệm 5x5:")
    print(test_matrix)
    
    # Chạy thuật toán
    tour, cost = solve(test_matrix, start_node=0)
    
    print(f"\nBắt đầu từ nút 0:")
    print(f"Lộ trình (Tour): {tour}")
    print(f"Chi phí (Cost): {cost}")
    
    # Kiểm tra kết quả mong đợi
    # 0 -> 1 (node 4)
    # 4 -> 3 (node 3)
    # 3 -> 1 (node 1)
    # 1 -> 5 (node 2)
    # Quay về: 2 -> 4 (node 0)
    # Tour: [0, 4, 3, 1, 2]
    # Cost: 1 + 3 + 1 + 5 + 4 = 14
    
    expected_tour = [0, 4, 3, 1, 2]
    expected_cost = 14
    
    assert tour == expected_tour
    assert cost == expected_cost
    
    print("\nKiểm tra logic NN thành công!")
    
    # Thử bắt đầu từ nút khác (ví dụ: nút 1)
    tour_1, cost_1 = solve(test_matrix, start_node=1)
    print(f"\nBắt đầu từ nút 1:")
    print(f"Lộ trình (Tour): {tour_1}")
    print(f"Chi phí (Cost): {cost_1}")
    # 1 -> 1 (node 3)
    # 3 -> 2 (node 2)
    # 2 -> 4 (node 0)
    # 0 -> 1 (node 4)
    # Tour: [1, 3, 2, 0, 4]
    # Cost: 1 + 2 + 4 + 1 + 6 = 14
    assert tour_1 == [1, 3, 2, 0, 4]
    assert cost_1 == 14
    print("Kiểm tra điểm bắt đầu khác thành công!")