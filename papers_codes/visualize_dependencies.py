import ast
import sys
import os
import importlib.util
from pyvis.network import Network
import networkx as nx

# 분석에서 제외할 외부 라이브러리 목록
EXCLUDED_MODULES = {
    'torch', 'matplotlib', 'numpy', 'pandas', 'scipy', 'sklearn',
    'tensorflow', 'keras', 'requests', 'seaborn', 'PIL'
}

class ProjectDependencyVisitor(ast.NodeVisitor):
    """
    하나의 파일을 분석하여 의존성 정보를 추출하는 방문자 클래스
    """
    def __init__(self, file_path, graph):
        self.graph = graph  # CodeVisualizer로부터 공유받는 전체 그래프
        self.file_name = os.path.basename(file_path)
        self.current_scope = self.file_name
        self.discovered_local_modules = []

        # 현재 파일 노드가 없으면 추가
        if not self.graph.has_node(self.file_name):
            self.graph.add_node(self.file_name, type='file', title=f"File: {self.file_name}", color='#FFD700', size=40)

    def visit_Import(self, node):
        for alias in node.names:
            module_name = alias.name
            if module_name.split('.')[0] in EXCLUDED_MODULES:
                continue
            self.discovered_local_modules.append(module_name)
            self.graph.add_node(module_name, type='module', title=f"Module: {module_name}", color='#ADD8E6')
            self.graph.add_edge(self.current_scope, module_name, label='imports')
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module_name = node.module
        if module_name and module_name.split('.')[0] not in EXCLUDED_MODULES:
            self.discovered_local_modules.append(module_name)
            self.graph.add_node(module_name, type='module', title=f"Module: {module_name}", color='#ADD8E6')
            self.graph.add_edge(self.current_scope, module_name, label='imports from')
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        func_name = node.name
        scoped_func_name = f"{self.file_name}::{func_name}"
        
        self.graph.add_node(scoped_func_name, type='function', title=f"Function: {func_name}\n(in {self.file_name})", color='#F08080')
        self.graph.add_edge(self.file_name, scoped_func_name, label='defines')

        original_scope = self.current_scope
        self.current_scope = scoped_func_name
        self.generic_visit(node)
        self.current_scope = original_scope

    def visit_Call(self, node):
        callee_name = self.get_call_name(node.func)
        if callee_name and callee_name.split('.')[0] not in EXCLUDED_MODULES:
            if not self.graph.has_node(callee_name):
                self.graph.add_node(callee_name, type='external_func', title=f"External/Built-in: {callee_name}", color='#D3D3D3')
            if self.current_scope != callee_name:
                self.graph.add_edge(self.current_scope, callee_name, label='calls')
        self.generic_visit(node)

    def get_call_name(self, func_node):
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            parent = self.get_call_name(func_node.value)
            return f"{parent}.{func_node.attr}" if parent else None
        return None


class CodeVisualizer:
    """
    프로젝트 전체의 코드 의존성을 재귀적으로 분석하고 시각화하는 클래스
    """
    def __init__(self, entry_point):
        self.entry_point = os.path.abspath(entry_point)
        self.project_root = os.path.dirname(self.entry_point)
        self.graph = nx.DiGraph()
        self.processed_files = set()

    def find_module_path(self, module_name, current_file_dir):
        """임포트된 모듈의 실제 파일 경로를 찾습니다."""
        # sys.path에 현재 파일 디렉토리를 임시 추가하여 상대 경로 임포트 해결
        original_sys_path = list(sys.path)
        sys.path.insert(0, current_file_dir)
        try:
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin and spec.origin.startswith(self.project_root):
                return spec.origin
        except (ModuleNotFoundError, AttributeError):
            return None
        finally:
            sys.path = original_sys_path
        return None

    def analyze(self):
        """진입점부터 시작하여 프로젝트 전체를 재귀적으로 분석합니다."""
        queue = [self.entry_point]

        while queue:
            current_file = queue.pop(0)
            if current_file in self.processed_files or not current_file.endswith('.py'):
                continue

            print(f"Analyzing: {os.path.basename(current_file)}...")
            self.processed_files.add(current_file)
            
            try:
                with open(current_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                tree = ast.parse(source_code)
                
                visitor = ProjectDependencyVisitor(current_file, self.graph)
                visitor.visit(tree)

                current_file_dir = os.path.dirname(current_file)
                for module_name in visitor.discovered_local_modules:
                    module_path = self.find_module_path(module_name, current_file_dir)
                    if module_path and module_path not in self.processed_files:
                        queue.append(module_path)

            except Exception as e:
                print(f"Error analyzing {current_file}: {e}")

    def generate_visualization(self):
        """분석된 그래프를 HTML 파일로 생성합니다."""
        if not self.graph.nodes:
            print("분석된 내용이 없습니다. 시각화를 생성할 수 없습니다.")
            return

        net = Network(height='1000px', width='100%', bgcolor='#222222', font_color='white', directed=True)
        net.from_nx(self.graph)
        net.set_options("""
        var options = {
          "nodes": {"font": {"size": 20, "strokeWidth": 2}, "scaling": {"min": 20, "max": 50}, "size": 35},
          "edges": {"width": 2, "font": {"size": 14, "align": "top", "strokeWidth": 3}, "arrows": {"to": { "enabled": true, "scaleFactor": 1.2 }}},
          "physics": {"barnesHut": {"gravitationalConstant": -60000, "centralGravity": 0.1, "springLength": 300}, "solver": "barnesHut"},
          "interaction": {"hover": true, "navigationButtons": true, "keyboard": true}
        }
        """)
        
        base_name = os.path.basename(self.entry_point)
        output_filename = f"project_dependencies_{os.path.splitext(base_name)[0]}.html"
        net.show(output_filename, notebook=False)
        print(f"\n성공! 현재 디렉토리에 '{output_filename}' 파일이 저장되었습니다.")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("사용법: python visualize_project.py <프로젝트_진입점_파일.py>")
    else:
        entry_file = sys.argv[1]
        visualizer = CodeVisualizer(entry_file)
        visualizer.analyze()
        visualizer.generate_visualization()