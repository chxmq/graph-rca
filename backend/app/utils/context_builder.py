from typing import Optional
from app.models.graph_data_models import DAGNode, DAG
from app.models.context_data_models import Context


class ContextBuilder:
    def __init__(self, dag: DAG) -> None:
        if not dag:
            raise ValueError("DAG is required to initialize ContextBuilder")
        self.dag = dag
        self.root_cause = None
        self.causal_chain = []

    def build_context(self) -> Context:
        """Build context from the DAG, extracting root cause and causal chain."""
        try:
            # Reset causal chain for fresh build
            self.causal_chain = []
            self.root_cause = self.dag.root_cause

            # Build O(n) lookup dict once — avoids O(n²) linear scan per recursive call
            node_lookup: dict[str, DAGNode] = {node.id: node for node in self.dag.nodes}

            # Use an iterative stack to avoid hitting Python's recursion limit on
            # large DAGs (default limit is 1 000 frames).
            stack = [self.dag.root_id]
            visited: set[str] = set()

            while stack:
                node_id = stack.pop()
                if not node_id or node_id in visited:
                    continue
                visited.add(node_id)

                node = node_lookup.get(node_id)
                if node is None:
                    continue

                self.causal_chain.append(node.log_entry.message)

                # Push children in reverse order so leftmost child is processed first
                for child_id in reversed(node.children):
                    if child_id not in visited:
                        stack.append(child_id)

            return Context(root_cause=self.root_cause, causal_chain=self.causal_chain)

        except Exception as e:
            raise RuntimeError(f"Failed to build context: {str(e)}")