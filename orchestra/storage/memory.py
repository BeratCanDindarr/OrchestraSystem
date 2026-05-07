"""Hierarchical Semantic memory (Project Atlas) with dependency tracking."""
from __future__ import annotations

import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional
from orchestra.providers.ollama import OllamaProvider

_ollama = OllamaProvider()

class SemanticMemory:
    """Project Atlas: Vector storage + Dependency Graph using SQLite."""
    
    def __init__(self, db_path: str = ".orchestra/memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Main embeddings table with summary support
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Robustly add missing columns to embeddings
            existing = {row[1] for row in conn.execute("PRAGMA table_info(embeddings)").fetchall()}
            if "summary" not in existing:
                conn.execute("ALTER TABLE embeddings ADD COLUMN summary TEXT")
            if "file_path" not in existing:
                conn.execute("ALTER TABLE embeddings ADD COLUMN file_path TEXT")

            # Dependency tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    source_path TEXT,
                    target_path TEXT,
                    relation_type TEXT,
                    PRIMARY KEY (source_path, target_path, relation_type)
                )
            """)
            conn.commit()

    def add(self, content: str, metadata: Dict[str, Any] = None, summary: str = None):
        """Add content and its summary to Atlas."""
        vector = _ollama.embed(summary or content[:1000])
        if not vector:
            return False
            
        vector_blob = np.array(vector, dtype=np.float32).tobytes()
        file_path = metadata.get("path") if metadata else None
        meta_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO embeddings (content, summary, vector, metadata, file_path) VALUES (?, ?, ?, ?, ?)",
                (content, summary, vector_blob, meta_json, file_path)
            )
            conn.commit()
        return True

    def add_relation(self, source: str, target: str, rel_type: str = "depends_on"):
        """Record a dependency between two project files."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO relations (source_path, target_path, relation_type) VALUES (?, ?, ?)",
                (source, target, rel_type)
            )
            conn.commit()

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content and include neighbor info."""
        query_vector = _ollama.embed(query)
        if not query_vector:
            return []
            
        query_vec = np.array(query_vector, dtype=np.float32)
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT content, summary, vector, metadata, file_path FROM embeddings")
            for content, summary, vector_blob, meta_json, file_path in cursor:
                vec = np.frombuffer(vector_blob, dtype=np.float32)
                # Handle cases where dimension might mismatch (rare but safe)
                if len(vec) != len(query_vec): continue
                similarity = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
                
                results.append({
                    "content": content,
                    "summary": summary,
                    "similarity": float(similarity),
                    "file_path": file_path,
                    "metadata": json.loads(meta_json) if meta_json else {}
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:limit]
        
        # Inject neighbors (dependencies) for top results
        for res in top_results:
            if res["file_path"]:
                res["dependencies"] = self.get_neighbors(res["file_path"])
                
        return top_results

    def get_neighbors(self, file_path: str) -> List[str]:
        """Find related files in the project atlas."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT target_path FROM relations WHERE source_path = ?", (file_path,)
            )
            return [row[0] for row in cursor.fetchall()]

def get_memory() -> SemanticMemory:
    import os
    os.makedirs(".orchestra", exist_ok=True)
    return SemanticMemory()
