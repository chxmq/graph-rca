from pydantic import BaseModel, Field
from typing import Optional
from app.models.graph_data_models import DAGNode,DAG
from app.models.parsing_data_models import LogChain
import logging

logger = logging.getLogger(__name__)

class GraphGenerator:
    def __init__(self,log_chain:LogChain) -> None:
        self.log_chain = log_chain
        self.dag_nodes = []
        self.root_id = None
        self.leaf_ids = []
        
    def generate_dag(self) -> DAG:
        """Generate a Directed Acyclic Graph (DAG) from the log chain"""
        try:
            self.dag_nodes = []
            self.root_id = None
            self.leaf_ids = []
            self.root_cause = None
            
            logger.info(f"[◆] Building DAG from {len(self.log_chain.log_chain)} log entries")
            for idx, log_entry in enumerate(self.log_chain.log_chain, 1):
                node = DAGNode(id=str(log_entry.timestamp),parent_id=None,children=[],log_entry=log_entry)
                self.dag_nodes.append(node)
                logger.info(f"  [►] Created node {idx}: {log_entry.level} at {log_entry.timestamp}")
                
            logger.info("[◆] Establishing parent-child relationships...")
            self._set_parent_child_relationships()
            logger.info("[◆] Identifying root and leaf nodes...")
            self._find_root_and_leaf_nodes()
            logger.info(f"[◆] Analyzing for root cause...")
            self.root_cause = self.find_root_cause()
            logger.info(f"[✓] Root cause determined: {self.root_cause}")
            
            return DAG(nodes=self.dag_nodes,root_id=self.root_id,leaf_ids=self.leaf_ids,root_cause=self.root_cause)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate DAG: {str(e)}")
        
    def _set_parent_child_relationships(self) -> None:
        """Set parent-child relationships between nodes based on timestamp order.
        Each node's parent is the immediately preceding node in time."""
        try:
            # Sort nodes by timestamp to ensure chronological order
            sorted_nodes = sorted(self.dag_nodes, key=lambda n: n.log_entry.timestamp)
            
            # Create sequential parent-child relationships
            for i in range(len(sorted_nodes) - 1):
                current_node = sorted_nodes[i]
                next_node = sorted_nodes[i + 1]
                
                # Set current node's child to be the next node
                current_node.children.append(next_node.id)
                # Set next node's parent to be the current node
                next_node.parent_id = current_node.id
            
            # Update the dag_nodes list to use sorted order
            self.dag_nodes = sorted_nodes
                        
        except Exception as e:
            raise RuntimeError(f"Failed to set parent-child relationships: {str(e)}")
        
    def _find_root_and_leaf_nodes(self) -> None:
        """Find the root and leaf nodes in the graph"""
        try:
            # Find root node (no parents)
            self.root_id = next(
                node.id for node in self.dag_nodes 
                if not node.parent_id
            )
            
            # Find leaf nodes (no children)
            nodes_with_children = set()
            for node in self.dag_nodes:
                nodes_with_children.update(node.children)
                
            self.leaf_ids = [
                node.id for node in self.dag_nodes
                if node.id not in nodes_with_children
            ]
            
        except Exception as e:
            raise RuntimeError(f"Failed to find root and leaf nodes: {str(e)}")

    def find_root_cause(self) -> str:
        """Find the root cause by analyzing the DAG structure with LLM.
        
        Uses the temporal ordering and log severity to identify the likely root cause,
        then uses LLM to synthesize a semantic explanation.
        """
        try:
            # Build context from DAG traversal
            causal_chain = self._build_causal_chain()
            
            # Identify candidate root causes (ERROR/CRITICAL logs near the start)
            candidates = self._identify_root_cause_candidates()
            
            if not candidates:
                # Fallback to first log if no clear candidates
                root_node = next(n for n in self.dag_nodes if n.id == self.root_id)
                return root_node.log_entry.message
            
            # Use LLM to analyze and synthesize root cause
            root_cause = self._analyze_with_llm(causal_chain, candidates)
            return root_cause
            
        except Exception as e:
            logger.warning(f"LLM analysis failed, falling back to heuristic: {e}")
            # Fallback: return first ERROR/CRITICAL message or first message
            return self._fallback_root_cause()
    
    def _build_causal_chain(self) -> str:
        """Build a textual representation of the causal chain from DAG."""
        chain_parts = []
        for i, node in enumerate(self.dag_nodes):
            entry = node.log_entry
            chain_parts.append(f"{i+1}. [{entry.level}] {entry.timestamp}: {entry.message}")
        return "\n".join(chain_parts)
    
    def _identify_root_cause_candidates(self) -> list:
        """Identify potential root cause candidates based on log severity and position."""
        candidates = []
        for i, node in enumerate(self.dag_nodes[:5]):  # Focus on first 5 logs (temporal proximity)
            entry = node.log_entry
            if entry.level.upper() in ["ERROR", "CRITICAL", "FATAL", "WARN", "WARNING"]:
                candidates.append({
                    "position": i + 1,
                    "level": entry.level,
                    "message": entry.message,
                    "timestamp": entry.timestamp
                })
        return candidates
    
    def _analyze_with_llm(self, causal_chain: str, candidates: list) -> str:
        """Use LLM to analyze the causal chain and determine root cause."""
        try:
            import ollama
            
            candidate_text = "\n".join([
                f"- Position {c['position']} [{c['level']}]: {c['message']}" 
                for c in candidates
            ])
            
            prompt = f"""Analyze this log cascade and identify the root cause of the incident.

Log Chain (chronological order):
{causal_chain}

Candidate root causes (early severe logs):
{candidate_text}

Based on the temporal ordering and log content, identify the root cause.
Respond with ONLY a concise root cause statement (1-2 sentences), nothing else."""

            # Use configured model or fallback
            model = "llama3.2:3b"
            response = ollama.generate(model=model, prompt=prompt, options={"temperature": 0.1})
            
            result = response.response.strip() if response and response.response else ""
            if result and len(result) > 10:
                return result
            
            # If LLM response is too short, use first candidate
            if candidates:
                return candidates[0]["message"]
            return self.dag_nodes[0].log_entry.message
            
        except ImportError:
            logger.warning("ollama not available, using heuristic fallback")
            return self._fallback_root_cause()
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return self._fallback_root_cause()
    
    def _fallback_root_cause(self) -> str:
        """Fallback heuristic: first ERROR/CRITICAL log or first log."""
        for node in self.dag_nodes:
            if node.log_entry.level.upper() in ["ERROR", "CRITICAL", "FATAL"]:
                return node.log_entry.message
        # Default to first log message
        if self.dag_nodes:
            return self.dag_nodes[0].log_entry.message
        return "Unable to determine root cause"
    
    def _find_root_cause_helper(self, node_id: str) -> str:
        """Legacy helper - kept for backwards compatibility."""
        return self._fallback_root_cause()
        