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

# --- Các hàm thành phần của Thuật toán Di truyền ---

def _calculate_fitness(cost: float) -> float:
    """Tính độ thích nghi: chi phí càng thấp, thích nghi càng cao."""
    # Thêm 1e-6 để tránh chia cho 0
    return 1.0 / (cost + 1e-6)

def _tournament_selection(population: List[List[int]], 
                          fitnesses: List[float], 
                          k: int) -> List[int]:
    """
    Chọn lọc Giải đấu (Tournament Selection).
    Chọn k cá thể ngẫu nhiên và trả về cá thể tốt nhất (thích nghi cao nhất).
    """
    # Lấy k chỉ số ngẫu nhiên từ quần thể
    indices = random.sample(range(len(population)), k)
    
    best_index = -1
    best_fitness = -np.inf
    
    for idx in indices:
        if fitnesses[idx] > best_fitness:
            best_fitness = fitnesses[idx]
            best_index = idx
            
    return population[best_index]

def _ordered_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Lai ghép có thứ tự (Ordered Crossover - OX1).
    Tạo ra một 'con' hợp lệ cho TSP.
    """
    num_cities = len(parent1)
    child = [None] * num_cities
    
    # 1. Chọn 2 điểm cắt ngẫu nhiên
    start, end = sorted(random.sample(range(num_cities), 2))
    
    # 2. Sao chép đoạn gen (segment) từ parent1 vào con
    segment = parent1[start : end + 1]
    child[start : end + 1] = segment
    segment_set = set(segment)
    
    # 3. Điền các gen còn thiếu từ parent2
    parent2_ptr = (end + 1) % num_cities
    child_ptr = (end + 1) % num_cities
    
    # Lặp cho đến khi 'con' được điền đầy
    while None in child:
        # Lấy gen từ parent2
        gene = parent2[parent2_ptr]
        
        # Nếu gen đó chưa có trong 'con' (từ segment của parent1)
        if gene not in segment_set:
            child[child_ptr] = gene
            child_ptr = (child_ptr + 1) % num_cities
            
        parent2_ptr = (parent2_ptr + 1) % num_cities
        
    return child

def _swap_mutation(tour: List[int]) -> List[int]:
    """
    Đột biến Hoán đổi (Swap Mutation).
    Hoán đổi 2 thành phố ngẫu nhiên trong lộ trình.
    """
    num_cities = len(tour)
    i, j = random.sample(range(num_cities), 2)
    
    mutated_tour = tour.copy()
    mutated_tour[i], mutated_tour[j] = mutated_tour[j], mutated_tour[i]
    return mutated_tour

# --- Hàm Solve chính ---

def solve(matrix: np.ndarray, 
          population_size: int = 100, 
          num_generations: int = 500, 
          mutation_rate: float = 0.01,
          elite_size: int = 5,
          tournament_k: int = 5) -> Tuple[List[int], int]:
    """
    Giải TSP bằng Thuật toán Di truyền (Genetic Algorithm).

    Tuân thủ "Interface" chuẩn: trả về (tour, cost).

    Args:
        matrix (np.ndarray): Ma trận khoảng cách (N x N).
        population_size (int): Số lượng cá thể (lộ trình) trong mỗi thế hệ.
        num_generations (int): Số thế hệ để tiến hóa.
        mutation_rate (float): Xác suất đột biến (ví dụ: 0.01 = 1%).
        elite_size (int): Số lượng cá thể 'tinh hoa' tốt nhất
                          được giữ lại ở mỗi thế hệ.
        tournament_k (int): Kích thước của 'giải đấu' khi chọn lọc.

    Returns:
        Tuple[List[int], int]:
            - tour (List[int]): Lộ trình tốt nhất tìm được.
            - cost (int): Chi phí của lộ trình tốt nhất.
    """
    num_cities = matrix.shape[0]
    
    # 1. Khởi tạo Quần thể
    population = []
    base_tour = list(range(num_cities))
    for _ in range(population_size):
        random.shuffle(base_tour)
        population.append(base_tour.copy())
        
    best_tour_overall = None
    best_cost_overall = np.inf

    # 2. Vòng lặp Tiến hóa (Generations)
    for gen in range(num_generations):
        
        # 3. Đánh giá (Fitness)
        costs = [calculate_tour_cost(tour, matrix) for tour in population]
        fitnesses = [_calculate_fitness(c) for c in costs]
        
        # 4. Elitism (Tinh hoa)
        # Sắp xếp quần thể theo chi phí (thấp đến cao)
        sorted_population = sorted(zip(population, costs), key=lambda x: x[1])
        
        # Cập nhật giải pháp tốt nhất toàn cục
        if sorted_population[0][1] < best_cost_overall:
            best_cost_overall = sorted_population[0][1]
            best_tour_overall = sorted_population[0][0].copy()
            
        # Tạo thế hệ mới, bắt đầu bằng các cá thể 'tinh hoa'
        new_population = [tour for tour, cost in sorted_population[:elite_size]]
        
        # 5. Lai ghép & Đột biến (lấp đầy phần còn lại của thế hệ mới)
        while len(new_population) < population_size:
            # Chọn lọc cha mẹ
            parent1 = _tournament_selection(population, fitnesses, tournament_k)
            parent2 = _tournament_selection(population, fitnesses, tournament_k)
            
            # Lai ghép
            child = _ordered_crossover(parent1, parent2)
            
            # Đột biến
            if random.random() < mutation_rate:
                child = _swap_mutation(child)
                
            new_population.append(child)
            
        # Thế hệ mới thay thế thế hệ cũ
        population = new_population

    return best_tour_overall, int(best_cost_overall)