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
        
    # 'tour'       : [0, 4, 1, 3, 2] (các nút hiện tại)
    # 'rolled_tour': [4, 1, 3, 2, 0] (các nút tiếp theo)
    # np.roll(tour, -1) dịch chuyển mảng sang trái 1 vị trí
    
    rolled_tour = np.roll(tour, -1)
    
    # Chúng ta lấy các chỉ số (current_node, next_node) từ ma trận
    # matrix[tour, rolled_tour] sẽ lấy:
    # matrix[0, 4], matrix[4, 1], matrix[1, 3], matrix[3, 2], matrix[2, 0]
    
    segment_costs = matrix[tour, rolled_tour]
    
    return np.sum(segment_costs)


# --- ĐÂY LÀ HÀM BỊ THIẾU MÀ BẠN CẦN THÊM VÀO ---

def calculate_gap(cost, opt_cost):
    """
    Tính toán % chênh lệch (Optimality Gap) so với chi phí tối ưu.
    
    Công thức: Gap % = ((cost - opt_cost) / opt_cost) * 100
    """
    # Xử lý trường hợp an toàn (ví dụ: opt_cost = 0)
    if opt_cost is None or opt_cost == 0:
        # Nếu không có chi phí tối ưu, không thể tính Gap
        return 0.0 
    
    if cost == opt_cost:
        return 0.0
        
    try:
        gap = ((cost - opt_cost) / opt_cost) * 100
        return round(gap, 3) # Làm tròn 3 chữ số
    except ZeroDivisionError:
        return 0.0