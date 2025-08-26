import ast
import sys
import os
import importlib.util
from pyvis.network import Network
import networkx as nx

class Symbol:
    """임포트된 심볼의 정보를 저장하는 클래스"""
    def __init__(self, name, source_module, source_path=None):
        self.name = name
        self.source_module = source_module
        self.source_path = source_path

class InternalProjectVisitor(ast.NodeVisitor):
    """
    프로젝트 내부의 의존성 정보만 추출하는 방문자 클래스
    """
    def __init__(self, visualizer, file_path, graph):
        self.visualizer = visualizer
        self.file_path = file_path
        self.graph = graph
        self.file_name = os.path.basename(file_path)
        self.current_scope = self.file_name
        self.import_map = {}

        if not self.graph.has_node(self.file_name):
            self.graph.add_node(self.file_name, type='file', title=f"File: {self.file_name}", color='#FFD700', size=40)

    def visit_Import(self, node):
        for alias in node.names:
            module_name = alias.name
            asname = alias.asname or module_name
            source_path = self.visualizer.resolve_module_path(module_name, self.file_path)
            # 심볼 테이블에는 추가하되, 로컬 모듈이 아니면 그래프에는 추가하지 않음
            self.import_map[asname] = Symbol(module_name, module_name, source_path)

    def visit_ImportFrom(self, node):
        module_name = node.module
        if not module_name:
            return
            
        source_path = self.visualizer.resolve_module_path(module_name, self.file_path)
        # 로컬 모듈을 임포트하는 관계만 그래프에 표시
        if source_path:
             target_file_name = os.path.basename(source_path)
             if self.graph.has_node(target_file_name):
                 self.graph.add_edge(self.file_name, target_file_name, label='imports from')

        for alias in node.names:
            original_name = alias.name
            asname = alias.asname or original_name
            self.import_map[asname] = Symbol(original_name, module_name, source_path)
            
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
        caller_node = self.current_scope
        callee_fqn = self.resolve_call_fqn(node.func)

        # 호출 대상이 프로젝트 내부 함수('::' 포함)인 경우에만 엣지를 추가
        if callee_fqn and "::" in callee_fqn:
            if caller_node != callee_fqn:
                self.graph.add_edge(caller_node, callee_fqn, label='calls')

        self.generic_visit(node)

    def resolve_call_fqn(self, node):
        """AST 노드를 분석하여 호출된 함수의 완전한 이름을 반환합니다."""
        if isinstance(node, ast.Name):
            func_name = node.id
            if func_name in self.import_map:
                symbol = self.import_map[func_name]
                if symbol.source_path: # 로컬 모듈에서 임포트된 함수
                    return f"{os.path.basename(symbol.source_path)}::{symbol.name}"
            # 현재 파일 내 함수 또는 내장 함수 (내장 함수는 필터링됨)
            return f"{self.file_name}::{func_name}"
        
        elif isinstance(node, ast.Attribute):
            parts = []
            curr = node
            while isinstance(curr, ast.Attribute):
                parts.insert(0, curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                base_obj_name = curr.id
                
                if base_obj_name in self.import_map:
                    symbol = self.import_map[base_obj_name]
                    if symbol.source_path:
                        # my_obj.method() -> my_module.py::method (간단한 형태)
                        func_name = parts[0]
                        return f"{os.path.basename(symbol.source_path)}::{func_name}"
        return None # 외부 라이브러리 호출은 None 반환

class CodeVisualizer:
    def __init__(self, entry_point):
        self.entry_point = os.path.abspath(entry_point)
        self.project_root = os.path.dirname(self.entry_point)
        self.graph = nx.DiGraph()
        self.processed_files = set()
        self.module_path_cache = {}

    def resolve_module_path(self, module_name, current_file_path):
        if module_name in self.module_path_cache:
            return self.module_path_cache[module_name]

        current_dir = os.path.dirname(current_file_path)
        original_sys_path = list(sys.path)
        sys.path.insert(0, self.project_root) # 프로젝트 루트를 우선 검색
        sys.path.insert(0, current_dir)
        try:
            spec = importlib.util.find_spec(module_name)
            # spec.origin이 프로젝트 루트 내에 있고 .py 파일인지 확인
            if spec and spec.origin and spec.origin.startswith(self.project_root) and spec.origin.endswith('.py'):
                path = os.path.abspath(spec.origin)
                self.module_path_cache[module_name] = path
                return path
        except Exception:
            pass
        finally:
            sys.path = original_sys_path
        
        self.module_path_cache[module_name] = None
        return None

    def analyze_project(self):
        # 1단계: 프로젝트 내 모든 로컬 파이썬 파일 찾기
        files_to_process = {self.entry_point}
        processed_for_discovery = set()
        
        queue = [self.entry_point]
        while queue:
            current_file = queue.pop(0)
            if current_file in processed_for_discovery or not os.path.exists(current_file):
                 continue
            processed_for_discovery.add(current_file)

            with open(current_file, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                module_to_find = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_to_find = alias.name
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module_to_find = node.module
                
                if module_to_find:
                    path = self.resolve_module_path(module_to_find, current_file)
                    if path and path not in files_to_process:
                        files_to_process.add(path)
                        queue.append(path)

        # 2단계: 찾은 모든 로컬 파일을 분석
        for file_path in files_to_process:
             print(f"Analyzing: {os.path.basename(file_path)}...")
             try:
                 with open(file_path, 'r', encoding='utf-8') as f:
                     source_code = f.read()
                 tree = ast.parse(source_code)
                 visitor = InternalProjectVisitor(self, file_path, self.graph)
                 visitor.visit(tree)
             except Exception as e:
                 print(f"Error analyzing {file_path}: {e}")
    
    def generate_visualization(self):
        if not self.graph.nodes:
            print("분석된 내용이 없습니다.")
            return

        net = Network(height='1000px', width='100%', bgcolor='#222222', font_color='white', directed=True)
        net.from_nx(self.graph)
        net.set_options("""
        var options = {
          "nodes": {"font": {"size": 20}, "size": 35},
          "edges": {"width": 2, "font": {"size": 14, "align": "top"}, "arrows": {"to": { "enabled": true, "scaleFactor": 1.2 }}},
          "physics": {"barnesHut": {"gravitationalConstant": -80000, "centralGravity": 0.1, "springLength": 400}},
          "interaction": {"hover": true, "navigationButtons": true, "keyboard": true}
        }
        """)
        
        base_name = os.path.basename(self.entry_point)
        output_filename = f"{os.path.splitext(base_name)[0]}_internal_dependencies.html"
        net.show(output_filename, notebook=False)
        print(f"\n성공! 현재 디렉토리에 '{output_filename}' 파일이 저장되었습니다.")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("사용법: python visualize_internal_project.py <프로젝트_진입점_파일.py>")
    else:
        entry_file = sys.argv[1]
        visualizer = CodeVisualizer(entry_file)
        visualizer.analyze_project()
        visualizer.generate_visualization()