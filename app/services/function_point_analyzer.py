"""
Function Point Analysis Service.

Calculates Function Points (FP) for software estimation using IFPUG methodology.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FunctionPointAnalyzer:
    """Analyzes text to extract function points and calculate estimates."""
    
    # Function Point weights (IFPUG standard)
    FP_WEIGHTS = {
        "EI": {"low": 3, "average": 4, "high": 6},  # External Input
        "EO": {"low": 4, "average": 5, "high": 7},  # External Output
        "EQ": {"low": 3, "average": 4, "high": 6},  # External Inquiry
        "ILF": {"low": 7, "average": 10, "high": 15},  # Internal Logical File
        "EIF": {"low": 5, "average": 7, "high": 10},  # External Interface File
    }
    
    # Value Adjustment Factors (VAF) components
    VAF_FACTORS = [
        "data_communications",
        "distributed_data_processing",
        "performance",
        "heavily_used_configuration",
        "transaction_rate",
        "online_data_entry",
        "end_user_efficiency",
        "online_update",
        "complex_processing",
        "reusability",
        "installation_ease",
        "operational_ease",
        "multiple_sites",
        "facilitate_change",
    ]
    
    def __init__(self, default_hourly_rate: float = 100.0):
        """
        Initialize Function Point Analyzer.
        
        Args:
            default_hourly_rate: Default hourly rate for cost estimation
        """
        self.default_hourly_rate = default_hourly_rate
    
    def analyze_text(self, text: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze text and tasks to extract function points.
        
        Args:
            text: Agent response or scenario text
            tasks: List of tasks from task distribution
            
        Returns:
            Function Point estimation dictionary
        """
        function_points = []
        
        # Extract function points from text
        fp_from_text = self._extract_function_points_from_text(text)
        function_points.extend(fp_from_text)
        
        # Extract function points from tasks
        fp_from_tasks = self._extract_function_points_from_tasks(tasks)
        function_points.extend(fp_from_tasks)
        
        # Calculate totals
        unadjusted_fp = sum(fp["fp_count"] for fp in function_points)
        
        # Calculate Value Adjustment Factor
        vaf = self._calculate_vaf(text, tasks)
        adjusted_fp = int(unadjusted_fp * vaf)
        
        # Estimate effort (using industry standard: 1 FP ≈ 8-10 hours)
        hours_per_fp = 9.0  # Average
        estimated_hours = adjusted_fp * hours_per_fp
        estimated_days = estimated_hours / 8.0  # Assuming 8-hour workdays
        estimated_cost = estimated_hours * self.default_hourly_rate
        
        return {
            "total_function_points": adjusted_fp,
            "unadjusted_fp": unadjusted_fp,
            "value_adjustment_factor": round(vaf, 2),
            "adjusted_fp": adjusted_fp,
            "function_points": function_points,
            "estimated_hours": round(estimated_hours, 1),
            "estimated_days": round(estimated_days, 1),
            "estimated_cost": round(estimated_cost, 2),
            "hourly_rate": self.default_hourly_rate,
        }
    
    def _extract_function_points_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract function points from text analysis."""
        function_points = []
        text_lower = text.lower()
        
        # Keywords for different function types
        patterns = {
            "EI": [
                r"(?:create|add|insert|submit|save|update|modify|edit|delete|remove)\s+(?:user|data|record|form|input)",
                r"(?:input|entry|form|submit|save)\s+(?:data|information|record)",
            ],
            "EO": [
                r"(?:generate|create|produce|export|output|report|display|show|render)\s+(?:report|output|data|information)",
                r"(?:dashboard|analytics|visualization|chart|graph)",
            ],
            "EQ": [
                r"(?:search|query|find|lookup|retrieve|fetch|get|read)\s+(?:data|information|record)",
                r"(?:filter|sort|list|view|display)\s+(?:data|information)",
            ],
            "ILF": [
                r"(?:database|data\s+store|storage|repository|persistent|save)\s+(?:user|data|information|record)",
                r"(?:internal|local)\s+(?:data|file|storage)",
            ],
            "EIF": [
                r"(?:external|api|integration|interface|import|sync)\s+(?:data|service|system)",
                r"(?:third.party|external\s+system|api\s+call)",
            ],
        }
        
        for fp_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    # Determine complexity based on context
                    context = text[max(0, match.start() - 50):match.end() + 50]
                    complexity = self._determine_complexity(context, fp_type)
                    
                    function_points.append({
                        "function_name": self._extract_function_name(match.group(), text),
                        "function_type": fp_type,
                        "complexity": complexity,
                        "fp_count": self.FP_WEIGHTS[fp_type][complexity],
                        "description": match.group(),
                    })
        
        # Deduplicate
        seen = set()
        unique_fps = []
        for fp in function_points:
            key = (fp["function_name"], fp["function_type"])
            if key not in seen:
                seen.add(key)
                unique_fps.append(fp)
        
        return unique_fps[:20]  # Limit to 20 function points
    
    def _extract_function_points_from_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract function points from task descriptions."""
        function_points = []
        
        task_keywords = {
            "EI": ["create", "add", "input", "entry", "submit", "save"],
            "EO": ["generate", "export", "report", "output", "display", "visualize"],
            "EQ": ["search", "query", "find", "retrieve", "filter", "list"],
            "ILF": ["store", "save", "persist", "database"],
            "EIF": ["integrate", "import", "sync", "api", "external"],
        }
        
        for task in tasks:
            task_desc = task.get("description", task.get("task_name", "")).lower()
            
            for fp_type, keywords in task_keywords.items():
                if any(keyword in task_desc for keyword in keywords):
                    complexity = self._determine_complexity(task_desc, fp_type)
                    function_points.append({
                        "function_name": task.get("task_name", "Unknown")[:50],
                        "function_type": fp_type,
                        "complexity": complexity,
                        "fp_count": self.FP_WEIGHTS[fp_type][complexity],
                        "description": task.get("description", "")[:200],
                    })
                    break  # One FP per task
        
        return function_points
    
    def _determine_complexity(self, context: str, fp_type: str) -> str:
        """Determine complexity level based on context."""
        context_lower = context.lower()
        
        # High complexity indicators
        high_indicators = [
            "complex", "advanced", "sophisticated", "multiple", "many",
            "integrated", "real-time", "dynamic", "automated", "ai", "ml",
        ]
        
        # Low complexity indicators
        low_indicators = [
            "simple", "basic", "single", "static", "manual", "straightforward",
        ]
        
        high_count = sum(1 for indicator in high_indicators if indicator in context_lower)
        low_count = sum(1 for indicator in low_indicators if indicator in context_lower)
        
        if high_count >= 2:
            return "high"
        elif low_count >= 2:
            return "low"
        else:
            return "average"
    
    def _extract_function_name(self, match_text: str, full_text: str) -> str:
        """Extract a meaningful function name from match."""
        # Try to find a noun or object name near the match
        words = match_text.split()
        if len(words) >= 2:
            return " ".join(words[:3]).title()
        return "Function"
    
    def _calculate_vaf(self, text: str, tasks: List[Dict[str, Any]]) -> float:
        """
        Calculate Value Adjustment Factor (VAF).
        
        VAF ranges from 0.65 to 1.35 based on 14 general system characteristics.
        """
        combined_text = text.lower() + " " + " ".join(
            t.get("description", "") for t in tasks
        ).lower()
        
        # Score each VAF factor (0-5 scale)
        vaf_scores = {}
        
        for factor in self.VAF_FACTORS:
            score = self._score_vaf_factor(factor, combined_text)
            vaf_scores[factor] = score
        
        # Calculate VAF: VAF = 0.65 + (sum of all scores) * 0.01
        total_score = sum(vaf_scores.values())
        vaf = 0.65 + (total_score * 0.01)
        
        return min(1.35, max(0.65, vaf))  # Clamp between 0.65 and 1.35
    
    def _score_vaf_factor(self, factor: str, text: str) -> int:
        """Score a VAF factor based on text analysis."""
        # Factor-specific keywords
        factor_keywords = {
            "data_communications": ["api", "network", "http", "websocket", "rest"],
            "distributed_data_processing": ["distributed", "microservice", "cluster"],
            "performance": ["performance", "speed", "fast", "optimize", "efficient"],
            "heavily_used_configuration": ["frequent", "high", "many", "scale"],
            "transaction_rate": ["transaction", "request", "throughput", "rate"],
            "online_data_entry": ["online", "real-time", "live", "instant"],
            "end_user_efficiency": ["user", "interface", "ui", "ux", "experience"],
            "online_update": ["update", "modify", "change", "edit"],
            "complex_processing": ["complex", "algorithm", "calculation", "processing"],
            "reusability": ["reuse", "component", "library", "module"],
            "installation_ease": ["install", "deploy", "setup", "configuration"],
            "operational_ease": ["operate", "maintain", "monitor", "manage"],
            "multiple_sites": ["multi", "distributed", "geographic", "location"],
            "facilitate_change": ["flexible", "adaptable", "configurable", "extensible"],
        }
        
        keywords = factor_keywords.get(factor, [])
        matches = sum(1 for keyword in keywords if keyword in text)
        
        # Score 0-5 based on keyword matches
        return min(5, matches)

