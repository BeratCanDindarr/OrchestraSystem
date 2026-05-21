from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from typing import List, Dict

from orchestra.storage.db import get_db

class UsageAnalytics:
    def __init__(self):
        pass

    def get_breakdown(self, period: str = "daily") -> List[Dict]:
        """Get cost and token breakdown."""
        conn = get_db()
        
        # Decide grouping based on SQLite date functions
        if period == "daily":
            group_expr = "date(created_at)"
        elif period == "monthly":
            group_expr = "strftime('%Y-%m', created_at)"
        elif period == "weekly":
            group_expr = "strftime('%Y-%W', created_at)"
        else:
            group_expr = "date(created_at)"

        query = f"""
            SELECT 
                {group_expr} as period_key,
                COUNT(run_id) as total_runs,
                SUM(total_cost_usd) as total_cost
            FROM runs
            GROUP BY period_key
            ORDER BY period_key DESC
            LIMIT 30
        """
        rows = conn.execute(query).fetchall()
        conn.close()
        
        return [dict(r) for r in rows]

    def export_csv(self, data: List[Dict]) -> str:
        if not data:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()
        
    def generate_heatmap(self, data: List[Dict]) -> str:
        """Simple text-based heatmap for terminal."""
        if not data:
            return "No data"
            
        max_cost = max(r["total_cost"] or 0 for r in data)
        if max_cost == 0:
            max_cost = 1
            
        lines = []
        for r in data:
            cost = r["total_cost"] or 0
            blocks = int((cost / max_cost) * 20)
            bar = "█" * blocks
            lines.append(f"{r['period_key']:12} | {bar:<20} | ${cost:.3f}")
            
        return "\n".join(lines)
