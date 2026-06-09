from app.models import DAGNode, DAG, LogChain
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

# Component / group fallback edges are looser than trace_id correlation —
# we only link to a prior node with the same component/group if it occurred
# within this window, to avoid stitching together unrelated events from a
# long-running module.
COMPONENT_FALLBACK_WINDOW = timedelta(seconds=30)

class GraphGenerator:
    def __init__(self, log_chain: LogChain) -> None:
        self.log_chain = log_chain
        self.dag_nodes = []
        self.root_id = None
        self.leaf_ids = []
        
    def generate_dag(self, analyse_root: bool = True) -> DAG:
        """Generate a causal timeline structure from the log chain."""
        try:
            self.dag_nodes = []
            self.root_id = None
            self.leaf_ids = []
            
            logger.info("[◆] Building DAG from %d log entries", len(self.log_chain.log_chain))
            for idx, log_entry in enumerate(self.log_chain.log_chain, 1):
                node = DAGNode(id=f"{log_entry.timestamp.isoformat()}-{idx}", parent_ids=[], children=[], log_entry=log_entry)
                self.dag_nodes.append(node)
                logger.debug("  [►] Created node %d: %s at %s", idx, log_entry.level, log_entry.timestamp)

            logger.info("[◆] Establishing parent-child relationships...")
            self._set_parent_child_relationships()
            logger.info("[◆] Identifying root and leaf nodes...")
            self._find_root_and_leaf_nodes()
            root_cause = None
            if analyse_root:
                logger.info("[◆] Analyzing for root cause...")
                root_cause = self.find_root_cause()
                logger.info("[✓] Root cause determined: %s", root_cause)

            return DAG(nodes=self.dag_nodes, root_id=self.root_id, leaf_ids=self.leaf_ids, root_cause=root_cause)

        except Exception as e:
            raise RuntimeError(f"Failed to generate DAG: {str(e)}") from e
        
    def _set_parent_child_relationships(self) -> None:
        """Set causal edges from explicit correlation keys.

        Nodes that share a trace_id or request_id are linked to the most
        recent prior node with the same key.  When neither key is present,
        component and group are used as fallbacks.  Nodes with no matching
        key become roots — they are not linked chronologically, so
        uncorrelated log lines degrade to a flat list of independent roots
        rather than a false causal chain.
        """
        try:
            sorted_nodes = sorted(self.dag_nodes, key=lambda n: n.log_entry.timestamp)
            last_by_trace: dict[str, DAGNode] = {}
            last_by_request: dict[str, DAGNode] = {}
            last_by_component: dict[str, DAGNode] = {}
            last_by_group: dict[str, DAGNode] = {}

            for current in sorted_nodes:
                candidate_parents: list[DAGNode] = []
                entry = current.log_entry

                for lookup, value in (
                    (last_by_trace, entry.trace_id),
                    (last_by_request, entry.request_id),
                ):
                    if value and value in lookup:
                        candidate_parents.append(lookup[value])

                if not candidate_parents and not entry.trace_id and not entry.request_id:
                    for lookup, value in (
                        (last_by_component, entry.component),
                        (last_by_group, entry.group),
                    ):
                        if not (value and value in lookup):
                            continue
                        prior = lookup[value]
                        # Both timestamps are tz-aware (enforced by validate_timestamp);
                        # the subtraction is safe.
                        if entry.timestamp - prior.log_entry.timestamp <= COMPONENT_FALLBACK_WINDOW:
                            candidate_parents.append(prior)

                # candidate_parents may contain the same node via multiple keys
                # (e.g. a node that matches both trace_id and component).
                # The `not in` guards prevent duplicate edges.
                for parent in candidate_parents[:3]:
                    if parent.id not in current.parent_ids:
                        current.parent_ids.append(parent.id)
                    if current.id not in parent.children:
                        parent.children.append(current.id)

                if entry.trace_id:
                    last_by_trace[entry.trace_id] = current
                if entry.request_id:
                    last_by_request[entry.request_id] = current
                if entry.component:
                    last_by_component[entry.component] = current
                if entry.group:
                    last_by_group[entry.group] = current

            self.dag_nodes = sorted_nodes

        except Exception as e:
            raise RuntimeError(f"Failed to set parent-child relationships: {str(e)}") from e

    def _find_root_and_leaf_nodes(self) -> None:
        """Find the root and leaf nodes in the graph"""
        try:
            root_nodes = [node for node in self.dag_nodes if not node.parent_ids]
            self.root_id = root_nodes[0].id if root_nodes else self.dag_nodes[0].id
            self.leaf_ids = [node.id for node in self.dag_nodes if not node.children]

        except Exception as e:
            raise RuntimeError(f"Failed to find root and leaf nodes: {str(e)}") from e

    def find_root_cause(self) -> str:
        """Find root cause using deterministic heuristics."""
        try:
            # Identify candidate root causes (ERROR/CRITICAL logs near the start)
            candidates = self._identify_root_cause_candidates()
            
            logger.debug("GraphGenerator: Found %d root cause candidates", len(candidates))
            for c in candidates:
                logger.debug("  - [%s] %s...", c['level'], c['message'][:60])
            
            if not candidates:
                logger.debug("GraphGenerator: No candidates, using first log as root cause")
                root_node = next(n for n in self.dag_nodes if n.id == self.root_id)
                return root_node.log_entry.message
            
            return candidates[0]["message"]

        except Exception as e:
            logger.warning("Root cause heuristic failed, using fallback: %s", e)
            return self._fallback_root_cause()
    
    def _build_causal_chain(self) -> str:
        """Build a textual representation of the causal chain from DAG."""
        chain_parts = []
        for i, node in enumerate(self.dag_nodes):
            entry = node.log_entry
            chain_parts.append(f"{i+1}. [{entry.level}] {entry.timestamp}: {entry.message}")
        return "\n".join(chain_parts)
    
    def _identify_root_cause_candidates(self) -> list:
        """Identify potential root cause candidates by severity.

        Only ERROR/CRITICAL/FATAL count as candidates.  WARN/WARNING are
        kept in the causal chain but never selected as the heuristic root
        cause — a "cache miss" warning preceding a real failure should not
        outrank the failure itself.
        """
        candidates = []
        for i, node in enumerate(self.dag_nodes):
            entry = node.log_entry
            if entry.level.upper() in ["ERROR", "CRITICAL", "FATAL"]:
                candidates.append({
                    "position": i + 1,
                    "level": entry.level,
                    "message": entry.message,
                    "timestamp": entry.timestamp.isoformat(),
                })
        return candidates
    
    def _fallback_root_cause(self) -> str:
        """Fallback heuristic: first ERROR/CRITICAL log or first log."""
        for node in self.dag_nodes:
            if node.log_entry.level.upper() in ["ERROR", "CRITICAL", "FATAL"]:
                return node.log_entry.message
        if self.dag_nodes:
            return self.dag_nodes[0].log_entry.message
        return "Unable to determine root cause"
