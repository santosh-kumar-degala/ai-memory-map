from __future__ import annotations

from .db import db


def fetch_graph(limit_nodes: int = 200, min_edge_weight: int = 1) -> dict:
    with db() as conn:
        nodes = conn.execute(
            """
            SELECT e.id, e.name, COALESCE(SUM(m.mentions), 0) AS mentions
            FROM entities e
            LEFT JOIN entity_mentions m ON m.entity_id = e.id
            GROUP BY e.id
            ORDER BY mentions DESC, e.name ASC
            LIMIT ?
            """,
            (limit_nodes,),
        ).fetchall()

        node_ids = [int(r["id"]) for r in nodes]
        if not node_ids:
            return {"nodes": [], "edges": []}

        placeholders = ",".join("?" for _ in node_ids)
        edges = conn.execute(
            f"""
            SELECT src_entity_id, dst_entity_id, weight
            FROM edges
            WHERE weight >= ?
              AND src_entity_id IN ({placeholders})
              AND dst_entity_id IN ({placeholders})
            ORDER BY weight DESC
            """,
            (min_edge_weight, *node_ids, *node_ids),
        ).fetchall()

    return {
        "nodes": [
            {"id": int(r["id"]), "label": str(r["name"]), "value": int(r["mentions"])}
            for r in nodes
        ],
        "edges": [
            {
                "from": int(r["src_entity_id"]),
                "to": int(r["dst_entity_id"]),
                "value": int(r["weight"]),
                "title": f"Co-occurred {int(r['weight'])}x",
            }
            for r in edges
        ],
    }

