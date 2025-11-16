import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import os
import sys
import time
import numpy as np

# --- 1. THIẾT LẬP SYS.PATH ĐỂ IMPORT DỰ ÁN ---
# Thêm thư mục gốc (cao hơn 'gui_app' một cấp) vào sys.path
# để import các module 'utils' và 'algorithms'
try:
    # Cách tiếp cận này hoạt động khi chạy dưới dạng script .py
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Cách này hoạt động trong môi trường tương tác (Jupyter, IPython)
    # Giả định notebook đang nằm trong thư mục con của dự án
    current_dir = os.getcwd()

project_root = os.path.normpath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Đã thêm '{project_root}' vào sys.path")

# --- 2. IMPORT CÁC MODULE CỦA DỰ ÁN ---
try:
    from utils import data_loader, evaluator, visualizer
    from algorithms import (
        brute_force, held_karp, nearest_neighbor, nearest_insertion,
        christofides, two_opt, simulated_annealing, genetic_algorithm,
        ant_colony, tabu_search
    )
except ImportError as e:
    print(f"LỖI: Không thể import module của dự án: {e}")
    print("Hãy đảm bảo bạn đang chạy file này từ trong cấu trúc dự án.")
    sys.exit(1)


class LogisticsPlannerApp(ctk.CTk):
    """
    Ứng dụng Trình Lập kế hoạch Giao vận (Logistics Planner)
    """

    def __init__(self):
        super().__init__()

        # --- Cài đặt Cửa sổ Chính ---
        self.title("Logistics Planner v1.0")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- Dữ liệu Trạng thái của Ứng dụng ---
        self.matrix = None
        self.coords = None
        self.opt_cost = 0
        self.problem_name = "Chưa tải"
        self.data_dir = os.path.join(project_root, 'data')
        self.output_dir = os.path.join(project_root, 'results')
        self.temp_image_path = os.path.join(self.output_dir, "temp_tour.png")

        # Lưu kết quả trước đó để so sánh
        self.previous_result = {} # {'algo_name': '...', 'total_cost': ...}

        # --- Ánh xạ Tên Thuật toán sang Hàm ---
        self.algo_map = {
            "Nearest Neighbor": nearest_neighbor.solve,
            "Nearest Insertion": nearest_insertion.solve,
            "2-Opt (from NN)": two_opt.solve,
            "Simulated Annealing": simulated_annealing.solve,
            "Genetic Algorithm": genetic_algorithm.solve,
            "Ant Colony (ACO)": ant_colony.solve,
            "Tabu Search": tabu_search.solve,
            "Christofides": christofides.solve,
            "Held-Karp (N<=21)": held_karp.solve,
            "Brute Force (N<=10)": brute_force.solve,
        }
        
        # --- Cấu hình Layout (2 cột) ---
        self.grid_columnconfigure(0, weight=1) # Cột điều khiển
        self.grid_columnconfigure(1, weight=3) # Cột kết quả
        self.grid_rowconfigure(0, weight=1)

        # --- 3. TẠO CÁC KHUNG (FRAMES) ---
        self.create_control_frame()
        self.create_results_frame()

    def create_control_frame(self):
        """Tạo khung điều khiển bên trái"""
        self.control_frame = ctk.CTkFrame(self, width=300)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # --- 1. Khu vực Tải Dữ liệu ---
        ctk.CTkLabel(self.control_frame, text="1. Tải Dữ liệu", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10, anchor="w", padx=10)
        
        self.load_button = ctk.CTkButton(self.control_frame, text="Tải Tệp Đơn hàng (.tsp)", command=self.load_problem)
        self.load_button.pack(pady=5, padx=10, fill="x")

        self.problem_label = ctk.CTkLabel(self.control_frame, text=f"Tệp: {self.problem_name}", text_color="gray")
        self.problem_label.pack(pady=5, padx=10, anchor="w")

        # --- 2. Khu vực Cài đặt Chi phí ---
        ctk.CTkLabel(self.control_frame, text="2. Cài đặt Chi phí", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10, anchor="w", padx=10)
        
        ctk.CTkLabel(self.control_frame, text="Chi phí mỗi Km (VND):").pack(anchor="w", padx=10)
        self.cost_km_entry = ctk.CTkEntry(self.control_frame, placeholder_text="10000")
        self.cost_km_entry.insert(0, "10000")
        self.cost_km_entry.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(self.control_frame, text="Lương cố định Tài xế (VND):").pack(anchor="w", padx=10)
        self.cost_driver_entry = ctk.CTkEntry(self.control_frame, placeholder_text="300000")
        self.cost_driver_entry.insert(0, "300000")
        self.cost_driver_entry.pack(pady=5, padx=10, fill="x")

        # --- 3. Khu vực Chọn Chiến lược ---
        ctk.CTkLabel(self.control_frame, text="3. Chọn Chiến lược Tối ưu", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10, anchor="w", padx=10)

        self.algo_combo = ctk.CTkComboBox(self.control_frame, values=list(self.algo_map.keys()))
        self.algo_combo.pack(pady=5, padx=10, fill="x")
        self.algo_combo.set("Nearest Neighbor") # Mặc định

        # --- 4. Nút Chạy ---
        self.run_button = ctk.CTkButton(self.control_frame, text="Tính toán Lộ trình", 
                                        command=self.run_calculation,
                                        font=ctk.CTkFont(size=14, weight="bold"),
                                        state="disabled") # Vô hiệu hóa cho đến khi tải tệp
        self.run_button.pack(pady=20, padx=10, fill="x", ipady=10)

    def create_results_frame(self):
        """Tạo khung kết quả bên phải"""
        self.results_frame = ctk.CTkFrame(self)
        self.results_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Tạo hệ thống Tab
        self.tab_view = ctk.CTkTabview(self.results_frame)
        self.tab_view.pack(expand=True, fill="both", padx=10, pady=10)

        self.tab_map = self.tab_view.add("Bản đồ Lộ trình")
        self.tab_cost = self.tab_view.add("Phân tích Chi phí")

        # --- Tab Bản đồ ---
        self.image_label = ctk.CTkLabel(self.tab_map, text="Vui lòng tải tệp và chạy tính toán để xem bản đồ.")
        self.image_label.pack(expand=True, fill="both")

        # --- Tab Phân tích Chi phí ---
        self.result_textbox = ctk.CTkTextbox(self.tab_cost, 
                                             font=ctk.CTkFont(family="Consolas", size=14),
                                             state="disabled") # Bắt đầu ở chế độ chỉ đọc
        self.result_textbox.pack(expand=True, fill="both")


    # --- 4. CÁC HÀM LOGIC (CALLBACKS) ---

    def load_problem(self):
        """Mở hộp thoại tệp và tải vấn đề TSP."""
        file_path = filedialog.askopenfilename(
            title="Chọn tệp đơn hàng (.tsp)",
            initialdir=os.path.join(self.data_dir, 'tsplib'),
            filetypes=(("TSP files", "*.tsp"), ("All files", "*.*"))
        )
        if not file_path:
            return

        # Lấy tên file (ví dụ: 'berlin52')
        self.problem_name = os.path.basename(file_path).replace('.tsp', '')
        
        try:
            problem_data = data_loader.load_problem(self.problem_name, self.data_dir)
            self.matrix = problem_data['matrix']
            self.coords = problem_data['coords']
            if problem_data['optimum_tour']:
                self.opt_cost = evaluator.calculate_tour_cost(problem_data['optimum_tour'], self.matrix)
            else:
                self.opt_cost = 0
            
            # Cập nhật GUI
            self.problem_label.configure(text=f"Tệp: {self.problem_name} (N={self.matrix.shape[0]})")
            self.run_button.configure(state="normal") # Kích hoạt nút chạy
            self.previous_result = {} # Xóa so sánh
            print(f"Đã tải {self.problem_name} thành công.")
        
        except Exception as e:
            self.problem_label.configure(text=f"Lỗi khi tải {self.problem_name}: {e}")
            self.run_button.configure(state="disabled")

    def run_calculation(self):
        """Chạy thuật toán đã chọn và cập nhật GUI."""
        if self.matrix is None:
            return

        # 1. Lấy thông tin từ GUI
        algo_name = self.algo_combo.get()
        algo_func = self.algo_map[algo_name]
        
        try:
            cost_per_km = float(self.cost_km_entry.get())
            driver_cost = float(self.cost_driver_entry.get())
        except ValueError:
            self.result_textbox.configure(state="normal")
            self.result_textbox.delete("1.0", "end")
            self.result_textbox.insert("end", "LỖI: Vui lòng nhập số hợp lệ cho chi phí.")
            self.result_textbox.configure(state="disabled")
            return

        # 2. Xử lý các thuật toán nặng (Cảnh báo & Kiểm tra)
        dim = self.matrix.shape[0]
        if algo_name == "Brute Force (N<=10)" and dim > 10:
            self.result_textbox.configure(state="normal")
            self.result_textbox.delete("1.0", "end")
            self.result_textbox.insert("end", f"LỖI: Brute Force không thể chạy với N={dim}.\nChỉ hỗ trợ N <= 10.")
            self.result_textbox.configure(state="disabled")
            return
        if algo_name == "Held-Karp (N<=21)" and dim > 21:
            self.result_textbox.configure(state="normal")
            self.result_textbox.delete("1.0", "end")
            self.result_textbox.insert("end", f"LỖI: Held-Karp không thể chạy với N={dim}.\nChỉ hỗ trợ N <= 21.")
            self.result_textbox.configure(state="disabled")
            return

        # 3. Chạy Thuật toán (và tính giờ)
        print(f"Đang chạy {algo_name}...")
        start_time = time.time()
        
        # Xử lý các thuật toán cần 'initial_tour'
        if algo_name in ["2-Opt (from NN)", "Simulated Annealing", "Tabu Search"]:
            # Chạy NN để lấy tour ban đầu
            nn_tour, _ = nearest_neighbor.solve(self.matrix, 0)
            tour, distance_cost = algo_func(self.matrix, initial_tour=nn_tour)
        else:
            tour, distance_cost = algo_func(self.matrix)
        
        exec_time = time.time() - start_time
        print(f"Hoàn thành trong {exec_time:.3f}s.")

        # 4. Tính toán Chi phí Kinh doanh
        fuel_cost = distance_cost * cost_per_km
        total_cost = fuel_cost + driver_cost
        gap = ((distance_cost - self.opt_cost) / self.opt_cost) * 100 if self.opt_cost > 0 else 0

        # 5. Cập nhật Tab "Phân tích Chi phí"
        self.update_cost_analysis(algo_name, distance_cost, total_cost, fuel_cost, driver_cost, gap, exec_time, cost_per_km)

        # 6. Cập nhật Tab "Bản đồ"
        self.update_map(algo_name, tour, distance_cost)
        
        # 7. Lưu kết quả để so sánh
        self.previous_result = {'algo_name': algo_name, 'total_cost': total_cost}
        
        # Tự động chuyển sang tab kết quả
        self.tab_view.set("Phân tích Chi phí")

    def update_cost_analysis(self, algo_name, dist, total_cost, fuel, driver, gap, time, cost_per_km):
        """Cập nhật hộp văn bản Phân tích Chi phí."""
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        
        output = []
        output.append(f"--- PHÂN TÍCH CHO: {algo_name.upper()} ---")
        output.append(f"Vấn đề: {self.problem_name} (N={self.matrix.shape[0]})")
        output.append("-" * 40)
        output.append(f"Tổng Khoảng cách Lộ trình: {dist:,.0f} km")
        output.append(f"Chi phí Nhiên liệu (@ {cost_per_km:,.0f} VND/km): {fuel:,.0f} VND")
        output.append(f"Lương cố định Tài xế: {driver:,.0f} VND")
        output.append("=" * 40)
        output.append(f"TỔNG CHI PHÍ CHUYẾN ĐI: {total_cost:,.0f} VND")
        output.append("=" * 40)
        
        # Thêm phần So sánh nếu có
        if self.previous_result:
            prev_name = self.previous_result['algo_name']
            prev_cost = self.previous_result['total_cost']
            savings = prev_cost - total_cost
            if savings > 0:
                output.append(f"\nSO SÁNH (vs {prev_name}):")
                output.append(f"  > TIẾT KIỆM: {savings:,.0f} VND")
            else:
                output.append(f"\nSO SÁNH (vs {prev_name}):")
                output.append(f"  > TỐN KÉM HƠN: {abs(savings):,.0f} VND")
                
        output.append("\n--- Chỉ số Thuật toán ---")
        output.append(f"Thời gian chạy: {time:.4f} giây")
        if self.opt_cost > 0:
            output.append(f"Khoảng cách (Gap): {gap:.2f}% so với tối ưu")
        
        self.result_textbox.insert("end", "\n".join(output))
        self.result_textbox.configure(state="disabled")

    def update_map(self, algo_name, tour, cost):
        """Vẽ lộ trình và hiển thị trong Tab Bản đồ."""
        print("Đang tạo hình ảnh bản đồ...")
        # 1. Sử dụng visualizer để lưu file ảnh tạm thời
        title = f"{algo_name} Tour ({self.problem_name})\nCost (Distance): {cost:.0f}"
        visualizer.plot_tour(
            tour=tour,
            coords=self.coords,
            title=title,
            save_path=self.temp_image_path
        )
        
        # 2. Mở file ảnh đó bằng PIL/Pillow
        img = Image.open(self.temp_image_path)
        
        # Lấy kích thước của tab chứa label để đảm bảo có kích thước đúng
        # ngay cả khi cửa sổ chưa được vẽ đầy đủ.
        tab_width = self.tab_map.winfo_width()
        tab_height = self.tab_map.winfo_height()
        # 3. Chuyển đổi nó sang CTkImage
        ctk_image = ctk.CTkImage(img, size=(tab_width, tab_height))
        
        # 4. Cập nhật label
        self.image_label.configure(image=ctk_image, text="") # Xóa văn bản mặc định
        print("Đã cập nhật bản đồ.")


if __name__ == "__main__":
    app = LogisticsPlannerApp()
    app.mainloop()