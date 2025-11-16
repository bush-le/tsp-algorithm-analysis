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

def solve(matrix: np.ndarray) -> Tuple[List[int], int]:
    """
    Giải TSP bằng thuật toán Held-Karp (Quy hoạch động Bitmask).

    Thuật toán này đảm bảo tìm ra giải pháp tối ưu.
    Độ phức tạp: O(N^2 * 2^N).
    Khả thi cho N <~ 20.

    Tuân thủ "Interface" chuẩn: trả về (tour, cost).

    Args:
        matrix (np.ndarray): Ma trận khoảng cách (N x N).

    Returns:
        Tuple[List[int], int]:
            - tour (List[int]): Lộ trình tối ưu (0-indexed).
            - cost (int): Chi phí tối ưu.
    """
    N = matrix.shape[0]
    
    # --- Xử lý các trường hợp cơ bản ---
    if N <= 1:
        return [0], 0
    if N == 2:
        return [0, 1], int(calculate_tour_cost([0, 1], matrix))

    # --- Khởi tạo ---
    # memo[mask][j] = chi phí tối thiểu để thăm các nút trong 'mask',
    # bắt đầu từ 0 và kết thúc tại 'j'.
    
    # (1 << N) là 2^N
    memo = np.full((1 << N, N), np.inf)
    
    # parent[mask][j] = nút 'i' (nút trước) dẫn đến trạng thái
    # (mask, j) tối ưu. Dùng để truy vết lộ trình.
    parent = np.full((1 << N, N), -1, dtype=int)
    
    # --- Trường hợp cơ sở ---
    # Chi phí để thăm {0} và kết thúc tại 0 là 0.
    start_node = 0
    memo[1 << start_node][start_node] = 0 # '1 << 0' là mask 00...01

    # --- Vòng lặp chính (Quy hoạch động) ---
    # Lặp qua tất cả các mặt nạ (subset) từ 1 đến 2^N - 1
    for mask in range(1, 1 << N):
        # Chỉ xem xét các tập hợp con chứa nút bắt đầu
        if not (mask & (1 << start_node)):
            continue
            
        # Lặp qua tất cả các nút 'j' (nút kết thúc)
        for j in range(N):
            # Nút 'j' phải nằm trong 'mask'
            if (mask & (1 << j)):
                
                # Tìm nút 'i' (nút trước)
                # 'i' cũng phải nằm trong 'mask' và i != j
                
                # Mặt nạ không có 'j'
                prev_mask = mask & ~(1 << j)
                if prev_mask == 0:
                    continue # Xảy ra khi mask = {j}, chỉ {0} là hợp lệ
                    
                for i in range(N):
                    if (prev_mask & (1 << i)):
                        # Tính chi phí mới
                        cost = memo[prev_mask][i] + matrix[i, j]
                        
                        if cost < memo[mask][j]:
                            memo[mask][j] = cost
                            parent[mask][j] = i # 'i' là nút trước của 'j'

    # --- Tìm kết quả cuối cùng ---
    # 'all_nodes_mask' là 11...11 (đã thăm tất cả N nút)
    all_nodes_mask = (1 << N) - 1
    min_cost = np.inf
    last_node = -1

    # Tìm chi phí tối ưu: min( cost(all, j) + dist(j, 0) )
    for j in range(1, N): # Bỏ qua j=0
        cost = memo[all_nodes_mask][j] + matrix[j, start_node]
        if cost < min_cost:
            min_cost = cost
            last_node = j # Nút cuối cùng (trước khi quay về 0)
            
    if last_node == -1: # Trường hợp N=2
         min_cost = memo[all_nodes_mask][1] + matrix[1, start_node]
         last_node = 1

    # --- Truy vết lộ trình ---
    tour = []
    current_node = last_node
    current_mask = all_nodes_mask

    for _ in range(N - 1):
        tour.append(current_node)
        prev_node = parent[current_mask][current_node]
        
        # Cập nhật mask và nút
        current_mask = current_mask & ~(1 << current_node)
        current_node = prev_node
        
    # Thêm nút bắt đầu
    tour.append(start_node)
    
    # Lộ trình đang bị ngược (ví dụ: [3, 2, 1, 0]), đảo lại
    best_tour = tour[::-1]
    
    return best_tour, int(min_cost)

# --- Ví dụ sử dụng (chỉ để kiểm tra nhanh) ---
if __name__ == "__main__":
    # 1. Sử dụng ma trận 4x4 (từ Wikipedia)
    test_matrix_4 = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    
    print("--- Chạy kiểm tra Held-Karp (N=4) ---")
    tour_4, cost_4 = solve(test_matrix_4)
    
    # Kết quả mong đợi (từ Brute Force): chi phí 80
    print(f"Tour tối ưu (HK): {tour_4}")
    print(f"Chi phí tối ưu (HK): {cost_4}")
    
    assert cost_4 == 80
    print("Kiểm tra (N=4) thành công!")

    # 2. Sử dụng ma trận 5x5
    test_matrix_5 = np.array([
        [0, 3, 4, 5, 1], # 0
        [3, 0, 5, 1, 6], # 1
        [4, 5, 0, 2, 7], # 2
        [5, 1, 2, 0, 3], # 3
        [1, 6, 7, 3, 0]  # 4
    ])
    
    print("\n--- Chạy kiểm tra Held-Karp (N=5) ---")
    tour_5, cost_5 = solve(test_matrix_5)
    
    # Kết quả mong đợi (từ Brute Force): chi phí 14
    print(f"Tour tối ưu (HK): {tour_5}")
    print(f"Chi phí tối ưu (HK): {cost_5}")
    
    assert cost_5 == 14
    # Kiểm tra lại chi phí bằng evaluator (để chắc chắn)
    verify_cost = calculate_tour_cost(tour_5, test_matrix_5)
    print(f"Chi phí (đã xác minh): {verify_cost}")
    assert cost_5 == verify_cost
    
    print("Kiểm tra (N=5) thành công!")