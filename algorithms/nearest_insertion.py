import numpy as np
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

def _find_best_insertion(tour: List[int], node_k: int, matrix: np.ndarray) -> int:
    """
    Tìm vị trí chèn tốt nhất (ít tăng chi phí nhất) cho 'node_k' vào 'tour'.

    Args:
        tour (List[int]): Lộ trình hiện tại.
        node_k (int): Nút mới cần chèn.
        matrix (np.ndarray): Ma trận khoảng cách.

    Returns:
        int: Chỉ số (index) trong 'tour' nơi 'node_k' nên được chèn vào *sau* nó.
             (Ví dụ: return 0 nghĩa là chèn vào sau tour[0])
    """
    min_cost_increase = np.inf
    best_index = 0
    
    # Lặp qua tất cả các cạnh (i, j) trong lộ trình hiện tại
    # Nếu tour = [0, 2, 4], các cạnh là (0, 2), (2, 4), và (4, 0)
    for i in range(len(tour)):
        node_i = tour[i]
        # Nút tiếp theo, xử lý vòng lặp
        node_j = tour[(i + 1) % len(tour)] 
        
        # Chi phí chèn: (i -> k) + (k -> j) - (i -> j)
        cost_increase = matrix[node_i, node_k] + matrix[node_k, node_j] - matrix[node_i, node_j]
        
        if cost_increase < min_cost_increase:
            min_cost_increase = cost_increase
            best_index = i # Chèn *sau* nút i
            
    return best_index

def solve(matrix: np.ndarray, start_node: int = 0) -> Tuple[List[int], int]:
    """
    Giải TSP bằng thuật toán Chèn Gần nhất (Nearest Insertion).

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
    if num_cities < 2:
        return list(range(num_cities)), 0
        
    # 1. Bắt đầu lộ trình với 'start_node'
    tour = [start_node]
    unvisited = np.ones(num_cities, dtype=bool)
    unvisited[start_node] = False
    
    # Tạo danh sách các ID nút chưa thăm
    unvisited_nodes = np.where(unvisited)[0].tolist()
    
    # 2. Xử lý trường hợp đặc biệt: thêm nút thứ 2 (gần nhất với điểm bắt đầu)
    if unvisited_nodes:
        # Tìm nút gần 'start_node' nhất
        second_node_dists = matrix[start_node, unvisited]
        # Đặt khoảng cách đến các nút đã thăm (chỉ 'start_node') là vô cùng
        distances_from_start = matrix[start_node].copy()
        distances_from_start[~unvisited] = np.inf
        
        second_node = np.argmin(distances_from_start)
        
        tour.append(second_node)
        unvisited[second_node] = False
        unvisited_nodes.remove(second_node)
        
    # 3. Lặp lại cho đến khi tất cả các nút được chèn
    while unvisited_nodes:
        # Bước "Nearest": Tìm nút 'k' chưa thăm gần nhất với *bất kỳ* nút nào
        # trong lộ trình hiện tại.
        
        # Lấy ma trận con của khoảng cách từ các nút CHƯA THĂM
        # đến các nút ĐÃ CÓ TRONG LỘ TRÌNH
        # Hàng: unvisited_nodes, Cột: tour
        sub_matrix = matrix[np.ix_(unvisited_nodes, tour)]
        
        # Tìm giá trị nhỏ nhất trong ma trận con này
        min_dist = np.min(sub_matrix)
        
        # Tìm chỉ số (hàng, cột) của giá trị nhỏ nhất đó
        # (row_idx, col_idx) tương ứng với chỉ số trong sub_matrix
        min_indices = np.where(sub_matrix == min_dist)
        # Lấy cặp đầu tiên tìm thấy
        row_idx, col_idx = min_indices[0][0], min_indices[1][0]
        
        # Lấy ID nút thực tế
        node_k = unvisited_nodes[row_idx] # Nút 'k' (gần nhất)
        
        # Bước "Insertion": Tìm vị trí chèn tốt nhất cho 'node_k'
        best_index = _find_best_insertion(tour, node_k, matrix)
        
        # Chèn 'node_k' vào sau 'best_index'
        tour.insert(best_index + 1, node_k)
        
        # Cập nhật danh sách chưa thăm
        unvisited[node_k] = False
        unvisited_nodes.remove(node_k)

    # Tính chi phí cuối cùng
    cost = calculate_tour_cost(tour, matrix)
    
    return tour, int(cost)

# --- Ví dụ sử dụng (chỉ để kiểm tra nhanh) ---
if __name__ == "__main__":
    # Sử dụng lại ma trận 5x5
    test_matrix = np.array([
        [0, 3, 4, 5, 1], # 0
        [3, 0, 5, 1, 6], # 1
        [4, 5, 0, 2, 7], # 2
        [5, 1, 2, 0, 3], # 3
        [1, 6, 7, 3, 0]  # 4
    ])
    
    print("--- Chạy kiểm tra Nearest Insertion ---")
    print("Ma trận thử nghiệm 5x5:")
    print(test_matrix)
    
    # Chạy thuật toán
    tour, cost = solve(test_matrix, start_node=0)
    
    print(f"\nBắt đầu từ nút 0:")
    print(f"Lộ trình (Tour): {tour}")
    print(f"Chi phí (Cost): {cost}")
    
    # Kiểm tra kết quả mong đợi
    # 1. Bắt đầu: tour = [0]
    # 2. Thêm nút thứ 2: Gần 0 nhất là 4 (cost 1). tour = [0, 4]
    # 3. Nút chưa thăm: [1, 2, 3]
    #    Nút trong tour: [0, 4]
    #    Sub_matrix (từ [1,2,3] đến [0,4]):
    #      (1): [3, 6]
    #      (2): [4, 7]
    #      (3): [5, 3]
    #    Min là 3, ở hai vị trí: (từ 1 đến 0) và (từ 3 đến 4).
    #    Giả sử np.where chọn (1, 0) trước -> node_k = 1.
    #    Chèn 1 vào [0, 4]:
    #      (0, 1, 4): C[0,1]+C[1,4]+C[4,0] = 3+6+1 = 10
    #      (0, 4, 1): C[0,4]+C[4,1]+C[1,0] = 1+6+3 = 10
    #      Giả sử _find_best_insertion chọn chèn sau 0 -> tour = [0, 1, 4]
    # 4. Nút chưa thăm: [2, 3]
    #    ... (tiếp tục logic) ...
    #
    # Chi phí tối ưu là 14. Nearest Insertion thường tìm thấy nó.
    
    assert cost == 14
    
    print("\nKiểm tra logic NI thành công!")