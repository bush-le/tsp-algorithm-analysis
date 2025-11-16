import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

def plot_tour(tour: List[int] | np.ndarray,
              coords: np.ndarray,
              title: str = "Lộ trình TSP",
              save_path: Optional[str] = None):
    """
    Vẽ một lộ trình (tour) TSP bằng Matplotlib.

    Hàm này vẽ các thành phố dưới dạng các điểm và lộ trình dưới dạng các đường
    nối các điểm đó theo thứ tự được chỉ định, bao gồm cả cạnh
    quay về điểm bắt đầu.

    Args:
        tour (List[int] | np.ndarray): Danh sách hoặc mảng các ID nút 
                                      (0-indexed) theo thứ tự.
        coords (np.ndarray): Mảng (N, 2) chứa tọa độ [x, y] của các nút.
        title (str): Tiêu đề cho biểu đồ.
        save_path (Optional[str]): Đường dẫn file để lưu hình ảnh 
                                   (ví dụ: 'results/my_tour.png').
                                   Nếu là None, biểu đồ sẽ được hiển thị.
    """
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
    # Lấy x, y gốc từ 'coords' để đảm bảo vẽ tất cả các điểm
    # (Mặc dù trong TSP chúng ta thường đi qua tất cả các điểm)
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ vào {save_path}")
        plt.close() # Đóng hình để tránh hiển thị trong kịch bản (script)
    else:
        plt.show() # Hiển thị biểu đồ tương tác