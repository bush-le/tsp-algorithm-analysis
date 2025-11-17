import numpy as np
import sys
import os
from typing import Tuple, List, Optional

# --- Import
try:
    from ..utils.evaluator import calculate_tour_cost
    # Giả định two_opt_improve được import từ module two_opt riêng
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from utils.evaluator import calculate_tour_cost
    # Cần thêm import two_opt nếu nó được dùng
    # Ví dụ: from algorithms.two_opt import solve as two_opt_solve 
    
# GHI CHÚ: Tôi sẽ giữ lại định nghĩa _two_opt_improve ở đây để code độc lập.
# Trong thực tế, bạn chỉ nên định nghĩa nó MỘT LẦN.


def _two_opt_improve(tour: List[int], matrix: np.ndarray, max_iter: int = 100) -> List[int]:
    """
    Cải thiện tour bằng 2-opt local search (đã dọn dẹp).
    """
    improved = tour.copy()
    changed = True
    iteration = 0
    
    # Tính toán total cost ban đầu (chỉ cần thiết cho vòng lặp, nhưng chúng ta dùng delta)
    n = len(tour)

    while changed and iteration < max_iter:
        changed = False
        iteration += 1
        
        # Chỉ lặp đến len(tour) - 1 để tránh lỗi index j+1
        for i in range(len(tour) - 1):
            for j in range(i + 2, len(tour)):
                
                # Cạnh cũ: (i, i+1) và (j, j+1)
                old_cost = (matrix[improved[i], improved[i+1]] + 
                            matrix[improved[j], improved[(j+1) % n]])
                
                # Cạnh mới: (i, j) và (i+1, j+1)
                new_cost = (matrix[improved[i], improved[j]] + 
                            matrix[improved[i+1], improved[(j+1) % n]])
                
                if new_cost < old_cost:
                    # Reverse segment [i+1:j+1]
                    improved[i+1:j+1] = list(reversed(improved[i+1:j+1]))
                    changed = True
                    break # First Improve
            
            if changed:
                break
        
    return improved


def _find_best_insertion_optimized(tour: List[int], node: int, matrix: np.ndarray) -> int:
    """
    Tìm vị trí chèn tốt nhất (delta cost) cho node vào tour.
    """
    min_increase = np.inf
    best_pos = 0
    tour_len = len(tour)
    
    for i in range(tour_len):
        current_i = tour[i]
        current_j = tour[(i + 1) % tour_len]
        
        # Cost increase = (i->node) + (node->j) - (i->j)
        increase = (matrix[current_i, node] + 
                    matrix[node, current_j] - 
                    matrix[current_i, current_j])
        
        if increase < min_increase:
            min_increase = increase
            best_pos = i
            
    return best_pos

def _find_nearest_unvisited(tour: List[int], 
                            unvisited: set, 
                            matrix: np.ndarray) -> int:
    """
    Tìm node chưa thăm gần nhất với bất kỳ node nào trong tour.
    """
    min_dist = np.inf
    nearest_node = None
    
    # --- ĐÃ SỬA LỖI LOGIC TÌM KIẾM ---
    for node in unvisited:
        # Tìm khoảng cách nhỏ nhất từ node hiện tại đến BẤT KỲ nút nào trong tour
        # Chỉ cần min distance, không cần min edge.
        dist_to_tour = min(matrix[node, tour_node] for tour_node in tour)
        
        if dist_to_tour < min_dist:
            min_dist = dist_to_tour
            nearest_node = node
            
    return nearest_node


def solve(matrix: np.ndarray, 
          start_node: int = 0,
          use_2opt: bool = False) -> Tuple[List[int], int]:
    """
    Giải TSP bằng Nearest Insertion.
    """
    num_cities = matrix.shape[0]
    
    # Edge cases
    if num_cities <= 2:
         tour = list(range(num_cities))
         cost = calculate_tour_cost(tour, matrix)
         return tour, int(cost)
    
    # 1. Khởi tạo tour với start_node
    tour = [start_node]
    unvisited = set(range(num_cities))
    unvisited.remove(start_node)
    
    # 2. Thêm node thứ 2 (nearest to start_node)
    second_node = min(unvisited, key=lambda x: matrix[start_node, x])
    tour.append(second_node)
    unvisited.remove(second_node)
    
    # 3. Main loop
    while unvisited:
        nearest_node = _find_nearest_unvisited(tour, unvisited, matrix)
        
        # 3b. Tìm vị trí chèn tốt nhất
        best_pos = _find_best_insertion_optimized(tour, nearest_node, matrix)
        
        # 3c. Chèn node
        tour.insert(best_pos + 1, nearest_node)
        
        # 3d. Update unvisited
        unvisited.remove(nearest_node)
        
    # 4. Optional: 2-opt improvement
    if use_2opt:
        tour = _two_opt_improve(tour, matrix)
        
    # 5. Tính cost
    cost = calculate_tour_cost(tour, matrix)
    
    return tour, int(cost)


def solve_multi_start(matrix: np.ndarray, 
                      num_starts: int = 5,
                      use_2opt: bool = False) -> Tuple[List[int], int]:
    """
    Chạy Nearest Insertion từ nhiều start nodes, chọn best.
    """
    num_cities = matrix.shape[0]
    
    if num_cities <= 2:
        return solve(matrix, 0, use_2opt)
    
    # Chọn start nodes đều nhau
    start_nodes = np.linspace(0, num_cities - 1, num_starts, dtype=int)
    
    best_tour = None
    best_cost = np.inf
    
    for start in start_nodes:
        # Gọi hàm solve NI cơ bản
        tour, cost = solve(matrix, start_node=start, use_2opt=use_2opt)
        
        if cost < best_cost:
            best_cost = cost
            best_tour = tour
            
    return best_tour, int(best_cost)