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
            # QUAN TRỌNG: Dùng negative weight để convert min → max
            G_odd.add_edge(u, v, weight=-matrix[u][v])
    
    # Tìm max weight matching (= min weight matching với negated weights)
    matching = nx.max_weight_matching(G_odd, maxcardinality=True, weight='weight')
    
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
    Giải TSP bằng thuật toán Christofides.
    
    Đảm bảo GAP ≤ 50% so với optimal (1.5-approximation).
    
    Steps:
    1. Tính Minimum Spanning Tree (MST)
    2. Tìm các đỉnh bậc lẻ trong MST
    3. Tìm Minimum Weight Perfect Matching cho các đỉnh bậc lẻ
    4. Kết hợp MST và Matching tạo thành Eulerian multigraph
    5. Tìm Eulerian circuit
    6. Shortcut để tạo Hamiltonian tour
    """
    num_cities = matrix.shape[0]
    
    # Edge case
    if num_cities <= 1:
        return list(range(num_cities)), 0
    
    # 1. Xây dựng MST (đảm bảo weights đúng)
    mst = _build_mst(matrix)
    
    # 2. Tìm các đỉnh bậc lẻ
    odd_nodes = _get_odd_degree_vertices(mst)
    
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
        # Fallback: trả về tour đơn giản
        tour = list(range(num_cities))
        cost = calculate_tour_cost(tour, matrix)
        return tour, int(cost)
    
    # 6. Tìm Eulerian circuit
    eulerian_circuit = _get_eulerian_circuit(multigraph, start_node=0)
    
    if not eulerian_circuit:
        # Fallback nếu không tìm được circuit
        tour = list(range(num_cities))
        cost = calculate_tour_cost(tour, matrix)
        return tour, int(cost)
    
    # 7. Shortcut để tạo Hamiltonian tour
    tour = _shortcut_tour(eulerian_circuit)
    
    # 8. Tính cost
    cost = calculate_tour_cost(tour, matrix)
    
    return tour, int(cost)