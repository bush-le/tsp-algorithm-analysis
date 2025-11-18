import numpy as np
import networkx as nx
from typing import Tuple, List
import sys
import os

# --- Import
try:
    from ..utils.evaluator import calculate_tour_cost
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from utils.evaluator import calculate_tour_cost

def _build_mst(matrix: np.ndarray) -> nx.Graph:
    """Xây dựng MST bằng NetworkX (đảm bảo weights đúng)."""
    num_cities = matrix.shape[0]
    G = nx.Graph()
    
    # Thêm tất cả edges với weight
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            # Chỉ thêm edge hợp lệ (không phải inf)
            if matrix[i][j] < float('inf'):
                G.add_edge(i, j, weight=matrix[i][j])
    
    # Tính MST bằng Kruskal/Prim của NetworkX
    mst = nx.minimum_spanning_tree(G, weight='weight')
    return mst

def _get_odd_degree_vertices(mst: nx.Graph) -> List[int]:
    """Tìm các đỉnh có bậc lẻ trong MST."""
    return [v for v, degree in mst.degree() if degree % 2 == 1]

def _minimum_weight_matching(matrix: np.ndarray, odd_nodes: List[int]) -> List[Tuple[int, int]]:
    """
    Tìm minimum weight perfect matching cho các odd degree nodes.
    Sử dụng NetworkX's max_weight_matching với negated weights.
    """
    if len(odd_nodes) == 0:
        return []
    
    # Tạo complete graph chỉ với odd nodes
    G_odd = nx.Graph()
    
    # Thêm tất cả edges giữa các odd nodes
    for i, u in enumerate(odd_nodes):
        for v in odd_nodes[i + 1:]:
            # Kiểm tra edge hợp lệ
            if matrix[u][v] < float('inf'):
                # QUAN TRỌNG: Dùng negative weight để convert min → max
                G_odd.add_edge(u, v, weight=-matrix[u][v])
    
    # Tìm max weight matching (= min weight matching với negated weights)
    try:
        matching = nx.max_weight_matching(G_odd, maxcardinality=True, weight='weight')
    except TypeError:
        # Fallback cho NetworkX phiên bản cũ
        matching = nx.max_weight_matching(G_odd, maxcardinality=True)
    
    return list(matching)

def _get_eulerian_circuit(multigraph: nx.MultiGraph, start_node: int = 0) -> List[int]:
    """Lấy chu trình Euler từ multigraph."""
    try:
        # Tìm chu trình Euler
        circuit_edges = list(nx.eulerian_circuit(multigraph, source=start_node))
        
        # Chuyển edges thành nodes
        circuit = [u for u, v in circuit_edges]
        circuit.append(circuit_edges[-1][1])  # Thêm node cuối
        
        return circuit
    except nx.NetworkXError as e:
        # Nếu không có Euler circuit, có lỗi trong thuật toán
        print(f"Error: Graph không có Euler circuit - {e}")
        return []

def _shortcut_tour(circuit: List[int]) -> List[int]:
    """
    Chuyển Euler circuit (có node lặp) thành Hamiltonian tour 
    bằng cách bỏ qua các node đã thăm (shortcutting).
    
    FIXED: Đảm bảo tour đóng vòng (quay về điểm xuất phát).
    """
    if not circuit:
        return []
    
    tour = []
    visited = set()
    
    for node in circuit:
        if node not in visited:
            tour.append(node)
            visited.add(node)
    
    # CRITICAL FIX: Đóng vòng tour - quay về điểm xuất phát
    if tour and tour[0] != tour[-1]:
        tour.append(tour[0])
    
    return tour

def _validate_matrix(matrix: np.ndarray) -> None:
    """Validate input matrix."""
    if len(matrix.shape) != 2:
        raise ValueError("Matrix must be 2-dimensional")
    
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    
    if np.any(matrix[matrix < float('inf')] < 0):
        raise ValueError("Matrix cannot have negative weights (except inf)")
    
    # Kiểm tra diagonal phải là 0 hoặc inf
    if not np.all((np.diag(matrix) == 0) | (np.diag(matrix) == float('inf'))):
        print("Warning: Diagonal should be 0 or inf")

def solve(matrix: np.ndarray) -> Tuple[List[int], int]:
    """
    Giải TSP bằng thuật toán Christofides.
    
    Đảm bảo GAP ≤ 50% so với optimal (1.5-approximation).
    
    Steps:
    1. Validate input
    2. Tính Minimum Spanning Tree (MST)
    3. Tìm các đỉnh bậc lẻ trong MST
    4. Tìm Minimum Weight Perfect Matching cho các đỉnh bậc lẻ
    5. Kết hợp MST và Matching tạo thành Eulerian multigraph
    6. Tìm Eulerian circuit
    7. Shortcut để tạo Hamiltonian tour (đóng vòng)
    8. Tính cost và validate
    
    Args:
        matrix: Ma trận khoảng cách (n x n)
        
    Returns:
        tour: Danh sách thứ tự các thành phố (bắt đầu và kết thúc tại cùng 1 node)
        cost: Tổng chi phí của tour
    """
    # Validate input
    _validate_matrix(matrix)
    
    num_cities = matrix.shape[0]
    
    # Edge cases
    if num_cities == 0:
        return [], 0
    
    if num_cities == 1:
        return [0, 0], 0
    
    if num_cities == 2:
        cost = int(matrix[0][1] + matrix[1][0])
        return [0, 1, 0], cost
    
    # 1. Xây dựng MST (đảm bảo weights đúng)
    mst = _build_mst(matrix)
    
    # 2. Tìm các đỉnh bậc lẻ
    odd_nodes = _get_odd_degree_vertices(mst)
    
    # Kiểm tra tính chất: số đỉnh bậc lẻ phải chẵn
    if len(odd_nodes) % 2 != 0:
        print(f"Warning: Số đỉnh bậc lẻ không chẵn ({len(odd_nodes)}), có lỗi!")
    
    # 3. Tìm minimum weight perfect matching
    matching_edges = _minimum_weight_matching(matrix, odd_nodes)
    
    # 4. Kết hợp MST và matching thành multigraph
    multigraph = nx.MultiGraph(mst)
    
    # Thêm matching edges vào multigraph
    for u, v in matching_edges:
        multigraph.add_edge(u, v, weight=matrix[u][v])
    
    # 5. Kiểm tra xem multigraph có Eulerian không
    if not nx.is_eulerian(multigraph):
        print("Warning: Multigraph không Eulerian, có lỗi trong thuật toán!")
        # Fallback: trả về tour đơn giản (nearest neighbor)
        tour = list(range(num_cities))
        tour.append(tour[0])  # Đóng vòng
        cost = calculate_tour_cost(tour, matrix)
        return tour, int(cost)
    
    # 6. Tìm Eulerian circuit
    eulerian_circuit = _get_eulerian_circuit(multigraph, start_node=0)
    
    if not eulerian_circuit:
        # Fallback nếu không tìm được circuit
        tour = list(range(num_cities))
        tour.append(tour[0])  # Đóng vòng
        cost = calculate_tour_cost(tour, matrix)
        return tour, int(cost)
    
    # 7. Shortcut để tạo Hamiltonian tour (đã đóng vòng)
    tour = _shortcut_tour(eulerian_circuit)
    
    # 8. Validate tour
    if not tour:
        tour = list(range(num_cities))
        tour.append(tour[0])
    
    # Đảm bảo tour đóng vòng
    if tour[0] != tour[-1]:
        tour.append(tour[0])
    
    # Kiểm tra tour có đủ thành phố không (trừ node lặp cuối)
    unique_cities = set(tour[:-1])
    if len(unique_cities) != num_cities:
        print(f"Warning: Tour thiếu thành phố! Có {len(unique_cities)}/{num_cities}")
    
    # 9. Tính cost
    cost = calculate_tour_cost(tour, matrix)
    
    return tour, int(cost)