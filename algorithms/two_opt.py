import numpy as np
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

def _calculate_delta(matrix: np.ndarray, 
                     tour: List[int], 
                     i: int, 
                     j: int) -> int:
    """
    Tính toán chênh lệch chi phí (delta) khi thực hiện 2-Opt swap.
    
    A = tour[i]
    B = tour[i+1]
    C = tour[j]
    D = tour[(j+1) % N] (nút tiếp theo, xử lý trường hợp vòng lặp)
    
    Chi phí cũ: (A -> B) + (C -> D)
    Chi phí mới: (A -> C) + (B -> D)
    
    Delta = Chi phí mới - Chi phí cũ
    """
    num_cities = len(tour)
    
    # Lấy các nút liên quan
    a = tour[i]
    b = tour[i + 1]
    c = tour[j]
    # (j+1) % num_cities xử lý trường hợp j là nút cuối cùng
    # và (j+1) là nút bắt đầu (tour[0])
    d = tour[(j + 1) % num_cities] 
    
    cost_old = matrix[a, b] + matrix[c, d]
    cost_new = matrix[a, c] + matrix[b, d]
    
    return int(cost_new) - int(cost_old)

def solve(matrix: np.ndarray, 
          initial_tour: Optional[List[int]] = None,
          max_iterations: int = 100) -> Tuple[List[int], int]:
    """
    Giải TSP bằng thuật toán 2-Opt, cải tiến một lộ trình ban đầu.

    Nếu không có 'initial_tour', một lộ trình đơn giản (0, 1, 2,...) sẽ được tạo.

    Tuân thủ "Interface" chuẩn: trả về (tour, cost).

    Args:
        matrix (np.ndarray): Ma trận khoảng cách (N x N).
        initial_tour (Optional[List[int]]): Lộ trình ban đầu để cải tiến.
        max_iterations (int): Số lần lặp tối đa qua toàn bộ lộ trình
                              để tìm kiếm cải tiến.

    Returns:
        Tuple[List[int], int]:
            - tour (List[int]): Lộ trình đã cải tiến.
            - cost (int): Chi phí của lộ trình đã cải tiến.
    """
    num_cities = matrix.shape[0]
    
    # Nếu không có tour ban đầu, tạo một tour đơn giản (0, 1, 2,...)
    if initial_tour is None:
        current_tour = list(range(num_cities))
    else:
        # Sao chép để không làm thay đổi bản gốc
        current_tour = initial_tour.copy()
        
    current_cost = calculate_tour_cost(current_tour, matrix)
    
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # l_range xác định chỉ số 'i' (cạnh đầu tiên)
        # Chúng ta bỏ qua nút cuối cùng (vì nó được xử lý bởi j+1)
        for i in range(num_cities - 1):
            # l_range xác định chỉ số 'j' (cạnh thứ hai)
            # j phải bắt đầu sau i (i+1) và bỏ qua các cạnh liền kề
            # (A-B-C) không thể swap (A,B) và (B,C)
            # Bắt đầu j từ i+2
            for j in range(i + 2, num_cities):
                
                # Xử lý trường hợp cạnh cuối cùng (ví dụ: N-1 và 0)
                # Chúng ta không swap cạnh đầu tiên (i=0) và cạnh cuối cùng
                if i == 0 and j == num_cities - 1:
                    continue 

                # Tính toán delta (chênh lệch chi phí)
                # (j+1) sẽ được xử lý vòng lặp bên trong _calculate_delta
                delta = _calculate_delta(matrix, current_tour, i, j)

                if delta < 0:
                    # Cải tiến được tìm thấy!
                    # Đảo ngược (reverse) đoạn giữa i+1 và j
                    # A - [B ... C] - D  =>  A - [C ... B] - D
                    
                    # Kỹ thuật slice:
                    # tour[i+1 : j+1] lấy đoạn [B...C] (bao gồm j)
                    # [::-1] đảo ngược nó
                    current_tour[i+1 : j+1] = current_tour[i+1 : j+1][::-1]
                    
                    # Cập nhật chi phí
                    current_cost += delta
                    improved = True
                    
                    # Thoát khỏi vòng lặp 'j' và 'i' để bắt đầu lại
                    # từ đầu với tour mới (Chiến lược First-Improve)
                    break 
            
            if improved:
                break # Thoát vòng lặp 'i'

    return current_tour, int(current_cost)