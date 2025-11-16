import numpy as np
from typing import List

def calculate_tour_cost(tour: List[int] | np.ndarray, 
                        matrix: np.ndarray) -> int | float:
    """
    Tính tổng chi phí của một lộ trình (tour) dựa trên ma trận khoảng cách.

    Lộ trình được cung cấp là một danh sách các nút, ví dụ [0, 1, 2].
    Hàm sẽ tính toán chi phí: 0->1 + 1->2 + 2->0 (quay lại điểm bắt đầu).

    Args:
        tour (List[int] | np.ndarray): Danh sách hoặc mảng các ID nút 
                                      (0-indexed) theo thứ tự của lộ trình.
        matrix (np.ndarray): Ma trận khoảng cách (N x N).

    Returns:
        int | float: Tổng chi phí của lộ trình.
    """
    if not isinstance(tour, np.ndarray):
        tour = np.array(tour)
        
    # 'tour'      : [0, 4, 1, 3, 2] (các nút hiện tại)
    # 'rolled_tour': [4, 1, 3, 2, 0] (các nút tiếp theo)
    # np.roll(tour, -1) dịch chuyển mảng sang trái 1 vị trí
    
    rolled_tour = np.roll(tour, -1)
    
    # Chúng ta lấy các chỉ số (current_node, next_node) từ ma trận
    # matrix[tour, rolled_tour] sẽ lấy:
    # matrix[0, 4], matrix[4, 1], matrix[1, 3], matrix[3, 2], matrix[2, 0]
    
    segment_costs = matrix[tour, rolled_tour]
    
    return np.sum(segment_costs)

# --- Ví dụ sử dụng (chỉ để kiểm tra nhanh) ---
if __name__ == "__main__":
    # Tạo một ma trận khoảng cách 5x5 đơn giản
    # Đây là ma trận từ ví dụ `utils/distance.py` (tam giác 3-4-5)
    # Mở rộng cho 5 nút
    #
    #   A B C D E
    # A 0 3 4 5 1
    # B 3 0 5 1 6
    # C 4 5 0 2 7
    # D 5 1 2 0 3
    # E 1 6 7 3 0
    #
    test_matrix = np.array([
        [0, 3, 4, 5, 1],
        [3, 0, 5, 1, 6],
        [4, 5, 0, 2, 7],
        [5, 1, 2, 0, 3],
        [1, 6, 7, 3, 0]
    ])
    
    # 1. Lộ trình đơn giản: 0 -> 1 -> 2 -> 0
    tour1 = [0, 1, 2]
    # Chi phí: matrix[0, 1] + matrix[1, 2] + matrix[2, 0]
    # Chi phí: 3 + 5 + 4 = 12
    cost1 = calculate_tour_cost(tour1, test_matrix)
    print(f"Lộ trình: {tour1}")
    print(f"Chi phí tính toán: {cost1} (Dự kiến: 12)")
    assert cost1 == 12
    
    # 2. Lộ trình đầy đủ: 0 -> 4 -> 3 -> 1 -> 2 -> 0
    tour2 = [0, 4, 3, 1, 2]
    # Chi phí: matrix[0, 4] + matrix[4, 3] + matrix[3, 1] + matrix[1, 2] + matrix[2, 0]
    # Chi phí: 1 + 3 + 1 + 5 + 4 = 14
    cost2 = calculate_tour_cost(tour2, test_matrix)
    print(f"\nLộ trình: {tour2}")
    print(f"Chi phí tính toán: {cost2} (Dự kiến: 14)")
    assert cost2 == 14

    print("\nKiểm tra Evaluator thành công!")