import numpy as np
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import Tuple, List
import sys
import os

# --- Import
# Xử lý sys.path để import từ thư mục 'utils'
try:
    from ..utils.evaluator import calculate_tour_cost
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from utils.evaluator import calculate_tour_cost

def _get_odd_degree_vertices(mst: nx.Graph) -> List[int]:
    """Tìm các đỉnh có bậc lẻ trong cây khung nhỏ nhất (MST)."""
    return [v for v, degree in mst.degree() if degree % 2 == 1]

def _get_eulerian_circuit(multigraph: nx.MultiGraph) -> List[int]:
    """Lấy chu trình Euler từ một đa đồ thị."""
    # Bắt đầu từ nút 0
    circuit_edges = list(nx.eulerian_circuit(multigraph, source=0))
    
    # circuit_edges là [(0, 1), (1, 5), ...]. 
    # Chúng ta chỉ cần các nút.
    circuit = [u for u, v in circuit_edges]
    # Thêm nút cuối cùng (nút bắt đầu)
    circuit.append(circuit_edges[-1][1])
    return circuit

def _shortcut_tour(circuit: List[int]) -> List[int]:
    """
    Chuyển đổi chu trình Euler (có lặp lại) thành lộ trình TSP
    (không lặp lại) bằng cách "đi đường tắt".
    """
    tour = []
    visited = set()
    
    for node in circuit:
        if node not in visited:
            tour.append(node)
            visited.add(node)
            
    return tour

def solve(matrix: np.ndarray) -> Tuple[List[int], int]:
    """
    Giải TSP bằng thuật toán xấp xỉ Christofides.

    Yêu cầu 'networkx' và 'scipy'.

    Tuân thủ "Interface" chuẩn: trả về (tour, cost).

    Args:
        matrix (np.ndarray): Ma trận khoảng cách (N x N).

    Returns:
        Tuple[List[int], int]:
            - tour (List[int]): Lộ trình TSP (0-indexed).
            - cost (int): Chi phí của lộ trình.
    """
    num_cities = matrix.shape[0]
    
    # 1. Tạo Đồ thị G đầy đủ
    # Chúng ta dùng ma trận tam giác trên vì networkx xử lý
    # đồ thị vô hướng một cách đối xứng.
    G = nx.from_numpy_array(np.triu(matrix), create_using=nx.Graph)
    
    # 2. Tính Cây khung nhỏ nhất (MST)
    # Dùng scipy để có hiệu suất tốt, sau đó chuyển sang NetworkX
    mst_scipy = minimum_spanning_tree(matrix)
    mst = nx.from_scipy_sparse_array(mst_scipy)
    
    # 3. Tìm các đỉnh bậc lẻ (O)
    odd_degree_nodes = _get_odd_degree_vertices(mst)
    
    # 4. Tìm Cặp ghép hoàn hảo trọng số tối thiểu (M)
    # Tạo đồ thị con chỉ chứa các đỉnh bậc lẻ
    odd_graph = G.subgraph(odd_degree_nodes)
    
    # Tìm cặp ghép hoàn hảo trọng số *tối thiểu*
    # Lưu ý: NetworkX gọi đây là `min_weight_matching`.
    # Chúng ta cần đảo ngược trọng số (biến chi phí thành lợi ích)
    # nếu dùng `max_weight_matching`, nhưng ở đây G là 
    # ma trận chi phí (distance), nên chúng ta cần tìm cặp ghép
    # sao cho tổng *trọng số* là nhỏ nhất.
    
    # NetworkX không có hàm `min_weight_perfect_matching`
    # trực tiếp cho đồ thị chung. 
    # nx.min_weight_matching là cho bipartite.
    # Chúng ta phải dùng một thủ thuật: tạo một đồ thị mới
    # với trọng số âm và tìm `max_weight_matching`
    
    inverted_matrix = nx.to_numpy_array(odd_graph)
    # Tạo ma trận "lợi ích" (utility)
    # Trọng số càng cao (ít âm) -> chi phí càng thấp (distance nhỏ)
    max_val = np.max(inverted_matrix[inverted_matrix > 0]) + 1
    utility_matrix = max_val - inverted_matrix
    np.fill_diagonal(utility_matrix, 0)
    
    inverted_graph = nx.from_numpy_array(utility_matrix)
    
    # Tìm cặp ghép hoàn hảo *trọng số tối đa* (max utility)
    # sẽ tương đương với *trọng số tối thiểu* (min distance)
    # `maxcardinality=True` đảm bảo nó là 'hoàn hảo'
    matching_edges = nx.max_weight_matching(inverted_graph, 
                                            maxcardinality=True)
    
    # Chuyển chỉ số từ `inverted_graph` (0, 1, 2,...)
    # về chỉ số nút gốc trong `G` (ví dụ: 3, 7, 12,...)
    node_map = list(odd_graph.nodes())
    original_matching_edges = [(node_map[u], node_map[v]) for u, v in matching_edges]

    # 5. Kết hợp MST và M (Tạo Đa đồ thị H)
    H = nx.MultiGraph(mst)
    H.add_edges_from(original_matching_edges)
    
    # 6. Tìm Chu trình Euler
    eulerian_circuit = _get_eulerian_circuit(H)
    
    # 7. "Đi đường tắt" (Shortcutting)
    tour = _shortcut_tour(eulerian_circuit)
    
    # Tính chi phí cuối cùng bằng evaluator của chúng ta
    cost = calculate_tour_cost(tour, matrix)
    
    return tour, int(cost)

# --- Ví dụ sửc dụng (chỉ để kiểm tra nhanh) ---
if __name__ == "__main__":
    # Yêu cầu cài đặt: pip install networkx scipy
    
    # 1. Sử dụng ma trận 4x4 (từ Wikipedia)
    test_matrix_4 = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    
    print("--- Chạy kiểm tra Christofides (N=4) ---")
    tour_4, cost_4 = solve(test_matrix_4)
    
    # Chi phí tối ưu là 80. Christofides là xấp xỉ.
    print(f"Tour (Christofides): {tour_4}")
    print(f"Cost (Christofides): {cost_4}")
    
    # Đối với N=4, MST = (0,1), (0,2), (0,3). Bậc lẻ: 1,2,3,0.
    # Matching sẽ phức tạp, nhưng kết quả phải gần 80
    assert cost_4 >= 80 # Phải lớn hơn hoặc bằng tối ưu
    
    # 2. Sử dụng ma trận 5x5
    test_matrix_5 = np.array([
        [0, 3, 4, 5, 1], # 0
        [3, 0, 5, 1, 6], # 1
        [4, 5, 0, 2, 7], # 2
        [5, 1, 2, 0, 3], # 3
        [1, 6, 7, 3, 0]  # 4
    ])
    
    print("\n--- Chạy kiểm tra Christofides (N=5) ---")
    tour_5, cost_5 = solve(test_matrix_5)
    
    # Chi phí tối ưu là 14.
    print(f"Tour (Christofides): {tour_5}")
    print(f"Cost (Christofides): {cost_5}")
    assert cost_5 >= 14
    
    print("\nKiểm tra Christofides hoàn tất (kết quả là xấp xỉ).")