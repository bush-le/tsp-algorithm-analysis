import numpy as np
import sys
import os
from typing import Tuple, List, Optional

# --- Import
try:
    from ..utils.evaluator import calculate_tour_cost
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from utils.evaluator import calculate_tour_cost

def _nearest_neighbor_single(matrix: np.ndarray, start_node: int) -> Tuple[List[int], float]:
    """
    Nearest Neighbor từ một start node cụ thể (tối ưu hiệu suất).
    
    Tối ưu:
    - Không copy array mỗi lần
    - Dùng set cho unvisited (O(1) lookup)
    - Trực tiếp tìm min thay vì mask
    """
    num_cities = matrix.shape[0]
    
    tour = [start_node]
    unvisited = set(range(num_cities))
    unvisited.remove(start_node)
    
    current = start_node
    
    while unvisited:
        # Tìm nearest unvisited neighbor
        nearest = min(unvisited, key=lambda city: matrix[current, city])
        
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    cost = calculate_tour_cost(tour, matrix)
    return tour, cost

def _two_opt_improve(tour: List[int], matrix: np.ndarray, max_iterations: int = 100) -> List[int]:
    """
    Cải thiện tour bằng 2-opt local search.
    
    2-opt: Thử đảo ngược mọi đoạn [i, j] và giữ nếu tốt hơn.
    """
    improved_tour = tour.copy()
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(len(tour) - 1):
            for j in range(i + 2, len(tour)):
                # Thử đảo ngược đoạn [i+1, j]
                new_tour = improved_tour[:i+1] + improved_tour[i+1:j+1][::-1] + improved_tour[j+1:]
                
                # So sánh cost (chỉ tính phần thay đổi)
                # Edge cũ: (i, i+1) + (j, j+1)
                # Edge mới: (i, j) + (i+1, j+1)
                n = len(tour)
                old_cost = (matrix[improved_tour[i], improved_tour[i+1]] + 
                           matrix[improved_tour[j], improved_tour[(j+1) % n]])
                new_cost = (matrix[improved_tour[i], improved_tour[j]] + 
                           matrix[improved_tour[i+1], improved_tour[(j+1) % n]])
                
                if new_cost < old_cost:
                    improved_tour = new_tour
                    improved = True
                    break
            
            if improved:
                break
    
    return improved_tour

def solve(matrix: np.ndarray, 
          start_node: Optional[int] = None,
          try_all_starts: bool = True,
          use_2opt: bool = True) -> Tuple[List[int], int]:
    """
    Giải TSP bằng Nearest Neighbor với cải tiến.
    
    Cải tiến:
    - Tối ưu hiệu suất (không copy array)
    - Thử tất cả start nodes (nếu enabled)
    - 2-opt improvement (nếu enabled)
    
    Args:
        matrix: Ma trận khoảng cách N×N
        start_node: Node bắt đầu cụ thể (None = thử tất cả)
        try_all_starts: Có thử tất cả start nodes không (khuyến nghị True)
        use_2opt: Có dùng 2-opt improvement không
        
    Returns:
        (best_tour, best_cost): Tour tốt nhất và chi phí
    """
    num_cities = matrix.shape[0]
    
    # Edge cases
    if num_cities == 0:
        return [], 0
    if num_cities == 1:
        return [0], 0
    if num_cities == 2:
        tour = [0, 1]
        cost = calculate_tour_cost(tour, matrix)
        return tour, int(cost)
    
    best_tour = None
    best_cost = np.inf
    
    # Xác định các start nodes cần thử
    if start_node is not None:
        # Chỉ thử 1 start node cụ thể
        start_nodes = [start_node]
    elif try_all_starts:
        # Thử tất cả start nodes
        start_nodes = range(num_cities)
    else:
        # Mặc định: chỉ thử node 0
        start_nodes = [0]
    
    # Thử mỗi start node
    for start in start_nodes:
        # Chạy nearest neighbor
        tour, cost = _nearest_neighbor_single(matrix, start)
        
        # Cải thiện bằng 2-opt (nếu enabled)
        if use_2opt:
            tour = _two_opt_improve(tour, matrix)
            cost = calculate_tour_cost(tour, matrix)
        
        # Update best
        if cost < best_cost:
            best_cost = cost
            best_tour = tour
    
    return best_tour, int(best_cost)

def solve_fast(matrix: np.ndarray, start_node: int = 0) -> Tuple[List[int], int]:
    """
    Version nhanh: Chỉ chạy từ 1 start node, không 2-opt.
    Dùng khi cần tốc độ, chấp nhận chất lượng thấp hơn.
    """
    return solve(matrix, start_node=start_node, try_all_starts=False, use_2opt=False)

def solve_best(matrix: np.ndarray) -> Tuple[List[int], int]:
    """
    Version tốt nhất: Thử tất cả starts + 2-opt.
    Dùng khi cần chất lượng cao, chấp nhận chậm hơn.
    """
    return solve(matrix, start_node=None, try_all_starts=True, use_2opt=True)