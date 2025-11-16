import numpy as np
import random
import math
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

def _get_neighbor(tour: List[int]) -> Tuple[List[int], Tuple[int, int]]:
    """
    Tạo một 'hàng xóm' bằng cách thực hiện 2-Opt swap ngẫu nhiên.
    Chọn 2 chỉ số ngẫu nhiên i và j, sau đó đảo ngược
    đoạn [i, j] hoặc [j, i].
    
    Trả về (tour_mới, (chỉ_số_1, chỉ_số_2))
    """
    num_cities = len(tour)
    
    # Chọn hai chỉ số ngẫu nhiên, không trùng nhau
    i, j = random.sample(range(num_cities), 2)
    
    # Đảm bảo i < j
    if i > j:
        i, j = j, i
        
    # Tạo tour mới bằng cách đảo ngược đoạn [i, j]
    # Ví dụ: tour = [0, 1, 2, 3, 4, 5]
    # i=2, j=4
    # tour[:i] = [0, 1]
    # tour[i:j+1][::-1] = [4, 3, 2]
    # tour[j+1:] = [5]
    # new_tour = [0, 1, 4, 3, 2, 5]
    
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    
    return new_tour, (i, j)

def solve(matrix: np.ndarray, 
          initial_tour: Optional[List[int]] = None,
          initial_temp: float = 1000.0,
          cooling_rate: float = 0.999,
          min_temp: float = 1.0,
          max_iterations_per_temp: int = 100) -> Tuple[List[int], int]:
    """
    Giải TSP bằng thuật toán Simulated Annealing (Luyện kim mô phỏng).

    Tuân thủ "Interface" chuẩn: trả về (tour, cost).

    Args:
        matrix (np.ndarray): Ma trận khoảng cách (N x N).
        initial_tour (Optional[List[int]]): Lộ trình ban đầu để cải tiến.
        initial_temp (float): Nhiệt độ ban đầu.
        cooling_rate (float): Tốc độ giảm nhiệt (ví dụ: 0.999).
        min_temp (float): Nhiệt độ tối thiểu để dừng.
        max_iterations_per_temp (int): Số lần lặp tại mỗi mức nhiệt.

    Returns:
        Tuple[List[int], int]:
            - tour (List[int]): Lộ trình tốt nhất tìm được.
            - cost (int): Chi phí của lộ trình tốt nhất.
    """
    num_cities = matrix.shape[0]
    
    # 1. Khởi tạo
    if initial_tour is None:
        current_tour = list(range(num_cities))
        random.shuffle(current_tour) # Bắt đầu với lộ trình ngẫu nhiên
    else:
        current_tour = initial_tour.copy()
        
    current_cost = calculate_tour_cost(current_tour, matrix)
    
    best_tour = current_tour.copy()
    best_cost = current_cost
    
    temperature = initial_temp
    
    # 2. Vòng lặp làm mát
    while temperature > min_temp:
        
        # Lặp N lần tại mỗi mức nhiệt
        for _ in range(max_iterations_per_temp):
            # 3. Tạo hàng xóm (dùng 2-Opt swap)
            # Chúng ta không cần _get_neighbor_ vì tính delta phức tạp hơn
            # Chúng ta sẽ tính toán delta trực tiếp
            
            # Chọn 2 chỉ số ngẫu nhiên i, j
            i, j = random.sample(range(num_cities), 2)
            if i > j: i, j = j, i
            if i == 0 and j == num_cities - 1: continue # Tránh cạnh (N-1, 0)
            
            # Tính toán delta chi phí
            # A = tour[i], B = tour[i+1]
            # C = tour[j], D = tour[(j+1) % N]
            a = current_tour[i]
            b = current_tour[(i + 1) % num_cities] # Phải dùng % để xử lý i=N-1
            c = current_tour[j]
            d = current_tour[(j + 1) % num_cities]
            
            cost_old = matrix[a, b] + matrix[c, d]
            cost_new = matrix[a, c] + matrix[b, d]
            delta_cost = cost_new - cost_old

            # 4. Quyết định chấp nhận
            if delta_cost < 0:
                # Cải tiến tốt hơn, luôn chấp nhận
                acceptance_prob = 1.0
            else:
                # Cải tiến tệ hơn, chấp nhận dựa trên xác suất
                if temperature == 0:
                    acceptance_prob = 0.0 # Tránh chia cho 0
                else:
                    acceptance_prob = math.exp(-delta_cost / temperature)
            
            # 5. Chấp nhận (hoặc không)
            if random.random() < acceptance_prob:
                # Áp dụng 2-Opt swap (đảo ngược)
                current_tour[i+1 : j+1] = current_tour[i+1 : j+1][::-1]
                # Cập nhật chi phí hiện tại
                current_cost += delta_cost
                
                # Cập nhật giải pháp tốt nhất (nếu cần)
                if current_cost < best_cost:
                    best_tour = current_tour.copy()
                    best_cost = current_cost
                    
        # 6. Giảm nhiệt độ
        temperature *= cooling_rate
        
    return best_tour, int(best_cost)