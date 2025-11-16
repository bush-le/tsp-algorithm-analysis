import numpy as np

def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Tính toán ma trận khoảng cách Euclidean từ một mảng tọa độ (N, 2).

    Sử dụng phép toán broadcasting của NumPy để tính toán vector hóa
    khoảng cách giữa tất cả các cặp điểm một cách hiệu quả.

    Khoảng cách được làm tròn theo tiêu chuẩn EUC_2D của TSPLIB.

    Args:
        coords (np.ndarray): Một mảng NumPy có hình dạng (N, 2),
                             trong đó N là số lượng thành phố và
                             mỗi hàng chứa tọa độ [x, y].

    Returns:
        np.ndarray: Một ma trận khoảng cách (N, N) đối xứng,
                    với các giá trị được làm tròn thành số nguyên.
    """
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)

    # coords (N, 2)
    # coords[:, np.newaxis, :] -> (N, 1, 2)
    # coords[np.newaxis, :, :] -> (1, N, 2)
    
    # Tính toán chênh lệch (diff) giữa mọi cặp điểm
    # Kết quả là một mảng (N, N, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    
    # Tính bình phương khoảng cách (d^2)
    # np.sum(..., axis=-1) cộng các bình phương của dx và dy
    # Kết quả là một mảng (N, N)
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Tính khoảng cách Euclidean (d)
    distances = np.sqrt(dist_sq)
    
    # Làm tròn theo tiêu chuẩn TSPLIB (EUC_2D)
    # np.round() làm tròn đến số nguyên gần nhất
    matrix = np.round(distances)
    
    return matrix.astype(int)

# --- Ví dụ sử dụng (chỉ để kiểm tra nhanh) ---
if __name__ == "__main__":
    # Đây là 3 điểm của một tam giác vuông 3-4-5
    test_coords = np.array([
        [0, 0],  # Điểm 0
        [3, 0],  # Điểm 1
        [0, 4]   # Điểm 2
    ])
    
    dist_matrix = calculate_distance_matrix(test_coords)
    
    print("Tọa độ các điểm:")
    print(test_coords)
    print("\nMa trận khoảng cách (đã làm tròn):")
    print(dist_matrix)
    
    # Kết quả mong đợi:
    # [[0, 3, 4],
    #  [3, 0, 5],
    #  [4, 5, 0]]
    
    assert dist_matrix[0, 1] == 3
    assert dist_matrix[0, 2] == 4
    assert dist_matrix[1, 2] == 5
    assert dist_matrix[1, 0] == 3
    print("\nKiểm tra thành công!")