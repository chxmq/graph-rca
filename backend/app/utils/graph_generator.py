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
        """Find the root cause of the issue"""
        try:
            root_cause = self._find_root_cause_helper(self.root_id)
            return root_cause
            
        except Exception as e:
            raise RuntimeError(f"Failed to find root cause: {str(e)}")
        
    def _find_root_cause_helper(self, node_id: str) -> str:
        """Helper function to find the root cause recursively"""
        try:
            node = next(node for node in self.dag_nodes if node.id == node_id)
            if not node.parent_id:
                return node.log_entry.message
            return self._find_root_cause_helper(node.parent_id)
            
        except Exception as e:
            raise RuntimeError(f"Failed to find root cause helper: {str(e)}")
        
    