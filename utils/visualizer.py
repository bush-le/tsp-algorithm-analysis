import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

def plot_tour(coords, tour, title, save_path=None):
    """
    Vẽ một lộ trình (tour) TSP bằng Matplotlib.

    Args:
        tour (List[int] | np.ndarray): Danh sách hoặc mảng các ID nút 
                                        (0-indexed) theo thứ tự.
        coords (np.ndarray | List): Mảng (N, 2) hoặc List các tuple (x, y) 
                                   chứa tọa độ của các nút.
        title (str): Tiêu đề cho biểu đồ.
        save_path (Optional[str]): Đường dẫn file để lưu hình ảnh 
                                   (ví dụ: 'results/my_tour.png').
    """
    
    # --- SỬA LỖI: Chuyển đổi coords (list of tuples) thành NumPy array ---
    # data_loader trả về một list, nhưng indexing (tour_coords = coords[tour])
    # yêu cầu coords phải là một NumPy array.
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    # --- KẾT THÚC SỬA LỖI ---
    
    if not isinstance(tour, np.ndarray):
        tour = np.array(tour)

    # Sắp xếp lại mảng tọa độ theo thứ tự của tour
    # tour_coords[i] sẽ là tọa độ của nút tour[i]
    tour_coords = coords[tour]
    
    # Để vẽ đường khép kín (quay lại điểm bắt đầu),
    # chúng ta cần thêm tọa độ của điểm đầu tiên vào cuối mảng
    closed_tour_coords = np.vstack([tour_coords, tour_coords[0]])
    
    # Lấy tọa độ x và y
    x = closed_tour_coords[:, 0]
    y = closed_tour_coords[:, 1]
    
    plt.figure(figsize=(10, 8))
    
    # Vẽ các đường (lộ trình)
    plt.plot(x, y, 'b-') # 'b-' là đường liền màu xanh
    
    # Vẽ các điểm (thành phố)
    plt.scatter(coords[:, 0], coords[:, 1], color='red', s=50, zorder=3)
    
    # Đánh dấu điểm bắt đầu (nút 0 của tour)
    start_node_coords = tour_coords[0]
    plt.scatter(start_node_coords[0], start_node_coords[1], 
                color='green', s=150, zorder=4, 
                label=f"Điểm bắt đầu (Nút {tour[0]})")
    
    plt.title(title, fontsize=16)
    plt.xlabel("Tọa độ X")
    plt.ylabel("Tọa độ Y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Đảm bảo tỷ lệ x-y bằng nhau
    
    if save_path:
        # Lưu file hình ảnh vào thư mục /results/
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu biểu đồ vào {save_path}")
        except Exception as e:
            print(f"LỖI: Không thể lưu hình ảnh vào {save_path}. Lỗi: {e}")
            
        plt.close() # Đóng hình để tránh hiển thị trong kịch bản (script)
    else:
        plt.show() # Hiển thị biểu đồ tương tác