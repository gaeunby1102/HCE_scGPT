"""
ontology.py
-----------
OntologyDAG: Cell Ontology(CL) 또는 Gene Ontology(GO)를 DAG로 표현.
실제 OBO 파일 없이 동작하는 mock builder 포함.
"""

from __future__ import annotations
import json
import warnings
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple


class OntologyDAG:
    """
    방향 비순환 그래프(DAG)로 표현된 생물학적 온톨로지.
    - 노드: 온톨로지 텀 (CL:0000001, GO:0008150 등)
    - 엣지 방향: child → parent (is_a 관계)
    """

    def __init__(self):
        self.nodes: Dict[str, str] = {}          # term_id -> term_name
        self.parents: Dict[str, Set[str]] = defaultdict(set)   # child -> parents
        self.children: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self._ancestor_cache: Dict[str, Set[str]] = {}
        self._depth_cache: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # 그래프 구성
    # ------------------------------------------------------------------

    def add_node(self, term_id: str, term_name: str = "") -> None:
        self.nodes[term_id] = term_name

    def add_edge(self, child: str, parent: str) -> None:
        """is_a 관계 추가: child is_a parent"""
        if child not in self.nodes:
            self.nodes[child] = child
        if parent not in self.nodes:
            self.nodes[parent] = parent
        self.parents[child].add(parent)
        self.children[parent].add(child)
        # 캐시 무효화
        self._ancestor_cache.clear()
        self._depth_cache.clear()

    # ------------------------------------------------------------------
    # 조회
    # ------------------------------------------------------------------

    def get_ancestors(self, term_id: str, include_self: bool = True) -> Set[str]:
        """term_id의 모든 조상 노드 집합 반환 (BFS)."""
        if term_id in self._ancestor_cache:
            result = self._ancestor_cache[term_id]
            return result if include_self else result - {term_id}

        visited: Set[str] = {term_id}
        queue = deque(self.parents.get(term_id, []))
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(self.parents.get(node, []))

        self._ancestor_cache[term_id] = visited
        return visited if include_self else visited - {term_id}

    def get_descendants(self, term_id: str, include_self: bool = True) -> Set[str]:
        """term_id의 모든 자손 노드 집합 반환 (BFS)."""
        visited: Set[str] = {term_id}
        queue = deque(self.children.get(term_id, []))
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(self.children.get(node, []))
        return visited if include_self else visited - {term_id}

    def get_depth(self, term_id: str) -> int:
        """루트까지의 최대 깊이 (루트=0)."""
        if term_id in self._depth_cache:
            return self._depth_cache[term_id]
        if not self.parents.get(term_id):
            self._depth_cache[term_id] = 0
            return 0
        depth = 1 + max(self.get_depth(p) for p in self.parents[term_id])
        self._depth_cache[term_id] = depth
        return depth

    def get_roots(self) -> List[str]:
        """부모가 없는 노드들(루트) 반환."""
        return [n for n in self.nodes if not self.parents.get(n)]

    def get_leaves(self) -> List[str]:
        """자식이 없는 노드들(리프) 반환."""
        return [n for n in self.nodes if not self.children.get(n)]

    def topological_sort(self) -> List[str]:
        """Kahn 알고리즘으로 위상 정렬 (루트 → 리프 순서)."""
        in_degree = {n: len(self.parents.get(n, set())) for n in self.nodes}
        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.children.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != len(self.nodes):
            warnings.warn("사이클이 감지되었습니다. DAG가 아닐 수 있습니다.")
        return order

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return (f"OntologyDAG(nodes={len(self.nodes)}, "
                f"edges={sum(len(v) for v in self.parents.values())})")


# ======================================================================
# 팩토리 함수들
# ======================================================================

def build_mock_cell_ontology() -> Tuple[OntologyDAG, Dict[str, int]]:
    """
    테스트용 Cell Ontology 서브그래프.
    뇌 싱글셀 분석에 자주 쓰이는 세포 유형 계층 구조.

    Cell (루트)
    ├── Neural cell
    │   ├── Neuron
    │   │   ├── Excitatory neuron
    │   │   │   ├── L2/3 IT neuron
    │   │   │   └── L5/6 NP neuron
    │   │   └── Inhibitory neuron
    │   │       ├── SST interneuron
    │   │       └── PV interneuron
    │   └── Glial cell
    │       ├── Astrocyte
    │       ├── Oligodendrocyte
    │       └── Microglia
    └── Immune cell
        ├── Lymphocyte
        └── Myeloid cell
    """
    dag = OntologyDAG()
    edges = [
        # (child, parent)
        ("neural_cell",          "cell"),
        ("immune_cell",          "cell"),
        ("neuron",               "neural_cell"),
        ("glial_cell",           "neural_cell"),
        ("excitatory_neuron",    "neuron"),
        ("inhibitory_neuron",    "neuron"),
        ("L23_IT",               "excitatory_neuron"),
        ("L56_NP",               "excitatory_neuron"),
        ("SST_interneuron",      "inhibitory_neuron"),
        ("PV_interneuron",       "inhibitory_neuron"),
        ("astrocyte",            "glial_cell"),
        ("oligodendrocyte",      "glial_cell"),
        ("microglia",            "glial_cell"),
        ("lymphocyte",           "immune_cell"),
        ("myeloid_cell",         "immune_cell"),
    ]
    node_names = {
        "cell": "Cell",
        "neural_cell": "Neural cell",
        "immune_cell": "Immune cell",
        "neuron": "Neuron",
        "glial_cell": "Glial cell",
        "excitatory_neuron": "Excitatory neuron",
        "inhibitory_neuron": "Inhibitory neuron",
        "L23_IT": "L2/3 IT neuron",
        "L56_NP": "L5/6 NP neuron",
        "SST_interneuron": "SST interneuron",
        "PV_interneuron": "PV interneuron",
        "astrocyte": "Astrocyte",
        "oligodendrocyte": "Oligodendrocyte",
        "microglia": "Microglia",
        "lymphocyte": "Lymphocyte",
        "myeloid_cell": "Myeloid cell",
    }
    for term_id, name in node_names.items():
        dag.add_node(term_id, name)
    for child, parent in edges:
        dag.add_edge(child, parent)

    # leaf 노드만 분류 대상으로 (term_to_idx)
    leaves = dag.get_leaves()
    term_to_idx = {term: i for i, term in enumerate(sorted(leaves))}
    return dag, term_to_idx


def build_mock_go_perturbation_ontology() -> Tuple[OntologyDAG, Dict[str, int]]:
    """
    테스트용 Gene Ontology 서브그래프 (섭동 결과 분류용).
    섭동이 영향을 주는 biological process의 계층 구조.

    Biological Process (루트)
    ├── Cellular process
    │   ├── Cell cycle
    │   │   ├── G1/S transition
    │   │   └── Mitosis
    │   └── Cell death
    │       ├── Apoptosis
    │       └── Necrosis
    ├── Metabolic process
    │   ├── Lipid metabolism
    │   └── Amino acid metabolism
    └── Immune process
        ├── Innate immunity
        │   ├── Cytokine signaling
        │   └── Phagocytosis
        └── Adaptive immunity
    """
    dag = OntologyDAG()
    edges = [
        ("cellular_process",      "biological_process"),
        ("metabolic_process",     "biological_process"),
        ("immune_process",        "biological_process"),
        ("cell_cycle",            "cellular_process"),
        ("cell_death",            "cellular_process"),
        ("g1s_transition",        "cell_cycle"),
        ("mitosis",               "cell_cycle"),
        ("apoptosis",             "cell_death"),
        ("necrosis",              "cell_death"),
        ("lipid_metabolism",      "metabolic_process"),
        ("aa_metabolism",         "metabolic_process"),
        ("innate_immunity",       "immune_process"),
        ("adaptive_immunity",     "immune_process"),
        ("cytokine_signaling",    "innate_immunity"),
        ("phagocytosis",          "innate_immunity"),
    ]
    node_names = {
        "biological_process": "Biological Process",
        "cellular_process": "Cellular Process",
        "metabolic_process": "Metabolic Process",
        "immune_process": "Immune Process",
        "cell_cycle": "Cell Cycle",
        "cell_death": "Cell Death",
        "g1s_transition": "G1/S Transition",
        "mitosis": "Mitosis",
        "apoptosis": "Apoptosis",
        "necrosis": "Necrosis",
        "lipid_metabolism": "Lipid Metabolism",
        "aa_metabolism": "Amino Acid Metabolism",
        "innate_immunity": "Innate Immunity",
        "adaptive_immunity": "Adaptive Immunity",
        "cytokine_signaling": "Cytokine Signaling",
        "phagocytosis": "Phagocytosis",
    }
    for term_id, name in node_names.items():
        dag.add_node(term_id, name)
    for child, parent in edges:
        dag.add_edge(child, parent)

    leaves = dag.get_leaves()
    term_to_idx = {term: i for i, term in enumerate(sorted(leaves))}
    return dag, term_to_idx


def load_ontology_from_json(path: str) -> OntologyDAG:
    """
    JSON 형식의 온톨로지 파일 로드.
    형식: {"nodes": [{"id": "CL:0000001", "name": "..."}],
            "edges": [{"child": "CL:0000002", "parent": "CL:0000001"}]}
    """
    with open(path) as f:
        data = json.load(f)
    dag = OntologyDAG()
    for node in data.get("nodes", []):
        dag.add_node(node["id"], node.get("name", ""))
    for edge in data.get("edges", []):
        dag.add_edge(edge["child"], edge["parent"])
    return dag
