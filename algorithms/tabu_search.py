import numpy as np
import random
import sys
import os
from typing import Tuple, List, Optional

# --- Import
# Xử lý sys.path để import từ thư mục 'utils'
try:
    from ..utils.evaluator import calculate_tour_cost
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from utils.evaluator import calculate_tour_cost

def _get_move_delta(matrix: np.ndarray, 
                    tour: List[int], 
                    i: int, 
                    j: int) -> int:
    """
    Tính delta chi phí cho một hoán đổi 2-Opt.
    Hoán đổi này đảo ngược đoạn [i+1, j].
    
    A = tour[i], B = tour[i+1]
    C = tour[j], D = tour[(j+1) % N]
    
    Delta = (A->C) + (B->D) - (A->B) - (C->D)
    """
    num_cities = len(tour)
    
    # Lấy 4 nút định nghĩa 2 cạnh
    a = tour[i]
    b = tour[(i + 1) % num_cities] # Phải dùng % để xử lý i=N-1
    c = tour[j]
    d = tour[(j + 1) % num_cities]
    
    cost_old = matrix[a, b] + matrix[c, d]
    cost_new = matrix[a, c] + matrix[b, d]
    
    return int(cost_new) - int(cost_old)

def solve(matrix: np.ndarray, 
          initial_tour: Optional[List[int]] = None,
          max_iterations: int = 1000, 
          tabu_tenure: int = 20) -> Tuple[List[int], int]:
    """
    Giải TSP bằng Tìm kiếm Cấm (Tabu Search).

    Args:
        matrix (np.ndarray): Ma trận khoảng cách (N x N).
        initial_tour (Optional[List[int]]): Lộ trình ban đầu.
        max_iterations (int): Tổng số lần lặp.
        tabu_tenure (int): "Nhiệm kỳ" (số lần lặp) mà một nước đi
                           bị cấm.

    Returns:
        Tuple[List[int], int]:
            - tour (List[int]): Lộ trình tốt nhất tìm được.
            - cost (int): Chi phí của lộ trình tốt nhất.
    """
    num_cities = matrix.shape[0]
    
    # 1. Khởi tạo
    if initial_tour is None:
        current_tour = list(range(num_cities))
        random.shuffle(current_tour) # Bắt đầu ngẫu nhiên
    else:
        current_tour = initial_tour.copy()
        
    current_cost = calculate_tour_cost(current_tour, matrix)
    
    best_tour = current_tour.copy()
    best_cost = current_cost
    
    # Tabu List: Dùng dict để lưu trữ 'nhiệm kỳ' cấm.
    # Key: (i, j) của nước đi. Value: số lần lặp còn lại.
    tabu_list = {}
    
    # 2. Vòng lặp chính
    for iteration in range(max_iterations):
        
        best_move = None
        best_delta = np.inf

        # 3. Khám phá tất cả các "hàng xóm" 2-Opt
        for i in range(num_cities - 1):
            for j in range(i + 2, num_cities):
                
                # 'move' được định nghĩa là 2 chỉ số (i, j)
                # Sắp xếp để (i, j) và (j, i) là như nhau
                move = tuple(sorted((i, j)))
                
                # Tính delta chi phí cho nước đi này
                delta = _get_move_delta(matrix, current_tour, i, j)
                
                # Kiểm tra xem nước đi có tốt hơn 'best_move' hiện tại không
                if delta < best_delta:
                    # Kiểm tra Tabu và Tiêu chí Chấp nhận
                    
                    is_tabu = move in tabu_list
                    
                    if not is_tabu:
                        # Nước đi không bị cấm, chấp nhận nó
                        best_move = move
                        best_delta = delta
                    else:
                        # Nước đi bị cấm, kiểm tra Tiêu chí Chấp nhận
                        # (Aspiration Criterion)
                        new_cost = current_cost + delta
                        if new_cost < best_cost:
                            # Tốt hơn best-ever, bỏ qua cấm!
                            best_move = move
                            best_delta = delta
                            
        # --- Cập nhật Tabu List ---
        # Giảm 'nhiệm kỳ' của tất cả các mục
        expired_moves = []
        for move, tenure in tabu_list.items():
            tenure -= 1
            if tenure <= 0:
                expired_moves.append(move)
            else:
                tabu_list[move] = tenure
                
        # Xóa các nước đi đã hết hạn
        for move in expired_moves:
            del tabu_list[move]

        # 4. Thực hiện nước đi tốt nhất
        if best_move is not None:
            i, j = best_move
            
            # Thực hiện 2-Opt swap (đảo ngược)
            current_tour[i+1 : j+1] = current_tour[i+1 : j+1][::-1]
            current_cost += best_delta
            
            # Thêm nước đi (ngược) vào Tabu List
            tabu_list[best_move] = tabu_tenure
            
            # Cập nhật giải pháp tốt nhất (nếu cần)
            if current_cost < best_cost:
                best_tour = current_tour.copy()
                best_cost = current_cost
        
        # Nếu không tìm thấy nước đi nào (hiếm), dừng
        if best_move is None:
            break
            
    return best_tour, int(best_cost)