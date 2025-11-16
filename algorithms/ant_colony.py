import numpy as np
import random
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

def _build_ant_tour(num_cities: int, 
                    probabilities: np.ndarray, 
                    start_node: int) -> List[int]:
    """Một con kiến xây dựng lộ trình dựa trên ma trận xác suất."""
    tour = [start_node]
    visited = np.zeros(num_cities, dtype=bool)
    visited[start_node] = True
    
    current_node = start_node
    
    for _ in range(num_cities - 1):
        # Lấy xác suất di chuyển từ nút hiện tại
        probs = probabilities[current_node].copy()
        
        # Ẩn các nút đã thăm
        probs[visited] = 0
        
        # Chuẩn hóa lại xác suất (để tổng bằng 1)
        prob_sum = np.sum(probs)
        if prob_sum == 0:
            # Bị kẹt (tất cả các nút còn lại đã thăm),
            # chọn ngẫu nhiên một nút chưa thăm (hiếm khi xảy ra)
            remaining = np.where(~visited)[0]
            next_node = random.choice(remaining)
        else:
            probs /= prob_sum
            
            # Chọn nút tiếp theo dựa trên xác suất
            next_node = np.random.choice(num_cities, p=probs)
        
        tour.append(next_node)
        visited[next_node] = True
        current_node = next_node
        
    return tour

def solve(matrix: np.ndarray, 
          num_ants: int = 20, 
          num_iterations: int = 100, 
          alpha: float = 1.0, 
          beta: float = 2.0,
          evaporation_rate: float = 0.5, 
          pheromone_deposit: float = 100.0) -> Tuple[List[int], int]:
    """
    Giải TSP bằng Tối ưu hóa Bầy kiến (Ant Colony Optimization).

    Args:
        matrix (np.ndarray): Ma trận khoảng cách (N x N).
        num_ants (int): Số lượng kiến trong mỗi vòng lặp.
        num_iterations (int): Số vòng lặp (thế hệ).
        alpha (float): Trọng số của pheromone.
        beta (float): Trọng số của tầm nhìn (1/distance).
        evaporation_rate (float): Tốc độ bay hơi (rho).
        pheromone_deposit (float): Lượng pheromone cơ sở (Q).

    Returns:
        Tuple[List[int], int]:
            - tour (List[int]): Lộ trình tốt nhất tìm được.
            - cost (int): Chi phí của lộ trình tốt nhất.
    """
    num_cities = matrix.shape[0]
    
    # --- 1. Khởi tạo ---
    
    # Pheromone: Bắt đầu với một lượng nhỏ trên tất cả các cạnh
    pheromone_matrix = np.ones((num_cities, num_cities)) * 1e-4
    
    # Visibility (Tầm nhìn): 1 / distance
    # Thêm epsilon (1e-6) để tránh chia cho 0 (khoảng cách 0 trên đường chéo)
    visibility_matrix = 1.0 / (matrix + 1e-6)
    np.fill_diagonal(visibility_matrix, 0)
    
    best_tour_overall = None
    best_cost_overall = np.inf

    # --- 2. Vòng lặp chính ---
    for _ in range(num_iterations):
        
        # --- 3. Xây dựng ma trận xác suất di chuyển ---
        # P(i,j) = (pheromone[i,j]^alpha) * (visibility[i,j]^beta)
        # Chúng ta chưa chuẩn hóa ở đây, sẽ chuẩn hóa trong _build_ant_tour
        prob_matrix = (pheromone_matrix ** alpha) * (visibility_matrix ** beta)
        
        all_tours = []
        all_costs = []

        # --- 4. Cho Bầy kiến xây dựng lộ trình ---
        for ant in range(num_ants):
            # Mỗi con kiến bắt đầu từ một thành phố ngẫu nhiên
            start_node = random.randint(0, num_cities - 1)
            
            tour = _build_ant_tour(num_cities, prob_matrix, start_node)
            cost = calculate_tour_cost(tour, matrix)
            
            all_tours.append(tour)
            all_costs.append(cost)
            
            # Cập nhật giải pháp tốt nhất (nếu cần)
            if cost < best_cost_overall:
                best_cost_overall = cost
                best_tour_overall = tour.copy()
                
        # --- 5. Cập nhật Pheromone ---
        
        # 5a. Bay hơi (Evaporation)
        pheromone_matrix *= (1.0 - evaporation_rate)
        
        # 5b. Gửi (Deposit)
        for tour, cost in zip(all_tours, all_costs):
            # Kiến gửi pheromone tỉ lệ nghịch với chi phí
            # (Q / Lk) - Mô hình Ant System (AS)
            deposit_amount = pheromone_deposit / cost
            
            for i in range(num_cities):
                j = (i + 1) % num_cities
                node_i = tour[i]
                node_j = tour[j]
                
                # Gửi pheromone lên cạnh (node_i, node_j)
                pheromone_matrix[node_i, node_j] += deposit_amount
                pheromone_matrix[node_j, node_i] += deposit_amount # Đồ thị vô hướng
                
    return best_tour_overall, int(best_cost_overall)

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
    
    print("--- Chạy kiểm tra Ant Colony Optimization (N=5) ---")
    
    # Chạy thuật toán
    tour, cost = solve(test_matrix,
                         num_ants=10,
                         num_iterations=50,
                         alpha=1.0, beta=2.0,
                         evaporation_rate=0.5,
                         pheromone_deposit=100)
    
    print(f"Tour tốt nhất (ACO): {tour}")
    print(f"Chi phí tốt nhất (ACO): {cost}")
    
    # Chi phí tối ưu là 14.
    assert cost == 14
    
    print("\nKiểm tra (N=5) thành công!")