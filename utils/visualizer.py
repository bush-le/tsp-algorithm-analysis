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

# --- Ví dụ sử dụng (chỉ để kiểm tra nhanh) ---
if __name__ == "__main__":
    # Tạo 5 tọa độ hình ngũ giác
    N = 5
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    coords = np.array([
        [np.cos(a), np.sin(a)] for a in angles
    ]) * 10
    coords = np.round(coords) # Làm tròn cho dễ nhìn
    
    # Thêm một điểm trung tâm
    coords = np.vstack([coords, [0, 0]]) # Nút 5
    
    print("Tọa độ (6 điểm):")
    print(coords)

    # Lộ trình 1: Chỉ đi 4 điểm
    # 0 -> 1 -> 3 -> 2 -> 0
    tour1 = [0, 1, 3, 2]
    plot_tour(tour1, coords, title="Kiểm tra Lộ trình 1 (4 điểm)")
    
    # Lộ trình 2: Đi tất cả 6 điểm
    # 0 -> 2 -> 4 -> 5 -> 3 -> 1 -> 0
    tour2 = [0, 2, 4, 5, 3, 1]
    
    # Kiểm tra lưu file (giả sử có thư mục /results)
    # Tạo đường dẫn lưu file an toàn
    import os
    # Giả sử chạy từ /utils, đi lên 1 cấp rồi vào /results
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    # Chuẩn hóa đường dẫn
    save_dir = os.path.normpath(save_dir)
    
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
            print(f"Đã tạo thư mục {save_dir}")
        except OSError:
            print(f"Không thể tạo thư mục {save_dir}. Chỉ hiển thị.")
            save_dir = None # Không lưu nữa
            
    save_path = None
    if save_dir:
        save_path = os.path.join(save_dir, "test_visualizer_tour.png")

    plot_tour(tour2, coords, 
              title="Kiểm tra Lộ trình 2 (6 điểm)", 
              save_path=save_path)