import numpy as np
import random
import sys
import os
from typing import Tuple, List

# --- Import
try:
    from ..utils.evaluator import calculate_tour_cost
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from utils.evaluator import calculate_tour_cost

# --- Các hàm thành phần ---

def _calculate_fitness(cost: float) -> float:
    """Tính độ thích nghi: chi phí càng thấp, thích nghi càng cao."""
    return 1.0 / (cost + 1e-6)

def _tournament_selection(population: List[List[int]], 
                          fitnesses: List[float], 
                          k: int) -> List[int]:
    """Chọn lọc Giải đấu (Tournament Selection)."""
    indices = random.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]

def _order_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Order Crossover (OX) - Crossover tốt nhất cho TSP.
    Giữ nguyên thứ tự tương đối từ parent2.
    """
    num_cities = len(parent1)
    
    # 1. Chọn một đoạn từ parent1
    start, end = sorted(random.sample(range(num_cities), 2))
    
    # 2. Copy đoạn đó vào child
    child = [None] * num_cities
    child[start:end+1] = parent1[start:end+1]
    segment_set = set(parent1[start:end+1])
    
    # 3. Lấy các gene còn lại từ parent2 theo thứ tự
    remaining_genes = [gene for gene in parent2 if gene not in segment_set]
    
    # 4. Điền vào child (bắt đầu từ vị trí sau segment)
    remaining_idx = 0
    for i in range(num_cities):
        if child[i] is None:
            child[i] = remaining_genes[remaining_idx]
            remaining_idx += 1
    
    return child

def _swap_mutation(tour: List[int]) -> List[int]:
    """Đột biến Hoán đổi: swap 2 thành phố ngẫu nhiên."""
    mutated = tour.copy()
    i, j = random.sample(range(len(tour)), 2)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

def _two_opt_mutation(tour: List[int]) -> List[int]:
    """Đột biến 2-opt: đảo ngược một đoạn tour."""
    mutated = tour.copy()
    i, j = sorted(random.sample(range(len(tour)), 2))
    mutated[i:j+1] = list(reversed(mutated[i:j+1]))
    return mutated

def _scramble_mutation(tour: List[int]) -> List[int]:
    """Đột biến Scramble: xáo trộn một đoạn nhỏ."""
    mutated = tour.copy()
    size = len(tour)
    length = random.randint(2, min(5, size // 3))  # Xáo trộn 2-5 cities
    start = random.randint(0, size - length)
    segment = mutated[start:start+length]
    random.shuffle(segment)
    mutated[start:start+length] = segment
    return mutated

def _calculate_diversity(population: List[List[int]]) -> float:
    """Tính diversity = tỷ lệ tour unique."""
    unique = len(set(tuple(tour) for tour in population))
    return unique / len(population)

def _greedy_init(matrix: np.ndarray, start_city: int) -> List[int]:
    """Tạo tour khởi tạo bằng nearest neighbor."""
    num_cities = matrix.shape[0]
    tour = [start_city]
    unvisited = set(range(num_cities)) - {start_city}
    
    current = start_city
    while unvisited:
        nearest = min(unvisited, key=lambda city: matrix[current][city])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return tour

# --- Hàm Solve chính ---

def solve(matrix: np.ndarray, 
          population_size: int = 100, 
          num_generations: int = 500, 
          mutation_rate: float = 0.15,
          elite_size: int = 5,
          tournament_k: int = 5,
          use_greedy_init: bool = True,
          diversity_threshold: float = 0.25) -> Tuple[List[int], int]:
    """
    Giải TSP bằng Genetic Algorithm với Order Crossover.
    
    Improvements:
    - Order Crossover (OX) thay vì PMX
    - 3 loại mutation (swap, 2-opt, scramble)
    - Greedy initialization cho một phần quần thể
    - Diversity monitoring với restart
    - Adaptive mutation rate
    
    Args:
        matrix: Ma trận khoảng cách
        population_size: Kích thước quần thể
        num_generations: Số thế hệ tối đa
        mutation_rate: Tỷ lệ đột biến (khuyến nghị 0.1-0.2)
        elite_size: Số cá thể ưu tú giữ lại
        tournament_k: Kích thước tournament
        use_greedy_init: Có dùng nearest neighbor cho init không
        diversity_threshold: Ngưỡng restart khi diversity thấp
    """
    num_cities = matrix.shape[0]
    
    # 1. Khởi tạo quần thể
    population = []
    
    # 1a. Một phần dùng greedy (nếu enabled)
    if use_greedy_init:
        greedy_count = min(10, population_size // 10)
        for i in range(greedy_count):
            start = i % num_cities
            tour = _greedy_init(matrix, start)
            population.append(tour)
    
    # 1b. Phần còn lại random
    while len(population) < population_size:
        tour = list(range(num_cities))
        random.shuffle(tour)
        population.append(tour)
    
    best_tour_overall = None
    best_cost_overall = np.inf
    no_improve_count = 0
    
    # 2. Vòng lặp tiến hóa
    for gen in range(num_generations):
        
        # 3. Đánh giá quần thể
        evaluated = []
        for tour in population:
            cost = calculate_tour_cost(tour, matrix)
            fitness = _calculate_fitness(cost)
            evaluated.append((tour, cost, fitness))
        
        # 4. Sắp xếp theo cost
        evaluated.sort(key=lambda x: x[1])
        
        # 5. Cập nhật best overall
        if evaluated[0][1] < best_cost_overall:
            best_cost_overall = evaluated[0][1]
            best_tour_overall = evaluated[0][0].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # 6. Kiểm tra diversity
        diversity = _calculate_diversity([x[0] for x in evaluated])
        
        # 7. Restart nếu mất diversity
        if diversity < diversity_threshold and gen > 50:
            print(f"[Gen {gen}] Diversity {diversity:.1%} < threshold, RESTARTING...")
            
            # Giữ top 10%, tạo mới 90%
            keep_count = max(5, population_size // 10)
            population = [x[0] for x in evaluated[:keep_count]]
            
            # Tạo mới với một nửa random, một nửa greedy
            while len(population) < population_size:
                if len(population) < population_size // 2:
                    # Random
                    tour = list(range(num_cities))
                    random.shuffle(tour)
                else:
                    # Greedy từ random start
                    start = random.randint(0, num_cities - 1)
                    tour = _greedy_init(matrix, start)
                population.append(tour)
            continue
        
        # 8. Adaptive mutation rate
        current_mutation_rate = mutation_rate
        if no_improve_count > 30:
            current_mutation_rate = min(0.4, mutation_rate * 2)
        elif no_improve_count > 50:
            current_mutation_rate = min(0.5, mutation_rate * 3)
        
        # 9. Elitism
        new_population = [x[0].copy() for x in evaluated[:elite_size]]
        
        # 10. Tạo thế hệ mới
        tours = [x[0] for x in evaluated]
        fitnesses = [x[2] for x in evaluated]
        
        while len(new_population) < population_size:
            # Selection
            parent1 = _tournament_selection(tours, fitnesses, tournament_k)
            parent2 = _tournament_selection(tours, fitnesses, tournament_k)
            
            # Crossover (Order Crossover - tốt nhất cho TSP)
            child = _order_crossover(parent1, parent2)
            
            # Mutation với 3 loại
            if random.random() < current_mutation_rate:
                mut_choice = random.random()
                if mut_choice < 0.5:
                    child = _swap_mutation(child)
                elif mut_choice < 0.85:
                    child = _two_opt_mutation(child)
                else:
                    child = _scramble_mutation(child)
            
            new_population.append(child)
        
        population = new_population
        
        # 11. Logging
        if (gen + 1) % 50 == 0 or gen < 5:
            print(f"[Gen {gen+1:3d}] Best={int(best_cost_overall):6d}, "
                  f"Current={int(evaluated[0][1]):6d}, "
                  f"Diversity={diversity:.1%}, "
                  f"MutRate={current_mutation_rate:.1%}, "
                  f"NoImprove={no_improve_count}")
    
    return best_tour_overall, int(best_cost_overall)