import numpy as np

def calculate_distance_matrix(coords: np.ndarray, edge_weight_type: str = "EUC_2D") -> np.ndarray:
    """
    Tính toán ma trận khoảng cách từ tọa độ (N, 2)
    dựa trên 'edge_weight_type' được chỉ định.
    
    Hỗ trợ: 'EUC_2D' và 'ATT'.
    """
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)

    # Tính bình phương khoảng cách (vector hóa)
    # (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # (N, N)
    dist_sq = np.sum(diff**2, axis=-1)
    
    if edge_weight_type == "EUC_2D":
        # Chuẩn EUC_2D: làm tròn số thực (float)
        distances = np.sqrt(dist_sq)
        matrix = np.round(distances)
        
    elif edge_weight_type == "ATT":
        # Chuẩn ATT: d = ceil(sqrt((x^2 + y^2) / 10.0))
        # np.ceil() là hàm "làm tròn lên" (ceiling)
        distances = np.sqrt(dist_sq / 10.0)
        matrix = np.ceil(distances)
        
    else:
        raise NotImplementedError(f"Kiểu trọng số '{edge_weight_type}' không được hỗ trợ để tính toán từ tọa độ.")

    return matrix.astype(int)