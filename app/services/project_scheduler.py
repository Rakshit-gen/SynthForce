"""
Project Scheduling Service.

Creates project timelines, schedules tasks, and identifies critical path.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ProjectScheduler:
    """Schedules project tasks and generates timelines."""
    
    def __init__(self, work_hours_per_day: float = 8.0):
        """
        Initialize Project Scheduler.
        
        Args:
            work_hours_per_day: Number of work hours per day
        """
        self.work_hours_per_day = work_hours_per_day
    
    def create_timeline(
        self,
        tasks: List[Dict[str, Any]],
        project_start_date: Optional[datetime] = None,
        agent_capacity: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Create project timeline with Gantt chart data.
        
        Args:
            tasks: List of tasks with dependencies and effort
            project_start_date: Start date for the project (datetime or ISO string)
            agent_capacity: Agent capacity in hours per day
            
        Returns:
            Project timeline dictionary
        """
        # Ensure project_start_date is a datetime object
        if project_start_date is None:
            project_start_date = datetime.utcnow()
        elif isinstance(project_start_date, str):
            try:
                project_start_date = datetime.fromisoformat(project_start_date.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                project_start_date = datetime.utcnow()
        elif not isinstance(project_start_date, datetime):
            project_start_date = datetime.utcnow()
        
        if agent_capacity is None:
            agent_capacity = {}
        
        # Convert tasks to scheduled tasks
        scheduled_tasks = self._schedule_tasks(
            tasks, project_start_date, agent_capacity
        )
        
        # Calculate critical path
        critical_path = self._calculate_critical_path(scheduled_tasks)
        
        # Calculate slack
        slack = self._calculate_slack(scheduled_tasks)
        
        # Find project end date
        end_dates = []
        for task in scheduled_tasks:
            end_date = task.get("end_date")
            if end_date:
                # Convert string to datetime if needed
                if isinstance(end_date, str):
                    try:
                        end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        continue
                if isinstance(end_date, datetime):
                    end_dates.append(end_date)
        
        project_end_date = max(end_dates) if end_dates else project_start_date
        
        # Calculate milestones
        milestones = self._identify_milestones(scheduled_tasks)
        
        total_duration = (project_end_date - project_start_date).days
        
        return {
            "project_start_date": project_start_date.isoformat(),
            "project_end_date": project_end_date.isoformat(),
            "total_duration_days": float(total_duration),
            "milestones": milestones,
            "tasks": scheduled_tasks,
            "critical_path": critical_path,
            "slack_days": slack,
        }
    
    def _schedule_tasks(
        self,
        tasks: List[Dict[str, Any]],
        start_date: datetime,
        agent_capacity: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Schedule tasks considering dependencies and resource constraints."""
        scheduled = []
        task_map = {task.get("task_id", f"task_{i}"): task for i, task in enumerate(tasks)}
        
        # Build dependency graph
        dependency_graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for task_id, task in task_map.items():
            deps = task.get("dependencies", [])
            in_degree[task_id] = len(deps)
            for dep in deps:
                dependency_graph[dep].append(task_id)
        
        # Topological sort to schedule tasks
        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])
        current_date = start_date
        agent_schedule = defaultdict(list)  # Track agent availability
        
        while queue:
            task_id = queue.popleft()
            task = task_map[task_id]
            
            # Calculate start date based on dependencies
            dep_end_dates = []
            for dep in task.get("dependencies", []):
                dep_task = next((t for t in scheduled if t["task_id"] == dep), None)
                if dep_task and dep_task.get("end_date"):
                    end_date = dep_task["end_date"]
                    # Convert string to datetime if needed
                    if isinstance(end_date, str):
                        try:
                            end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        except (ValueError, AttributeError):
                            continue
                    if isinstance(end_date, datetime):
                        dep_end_dates.append(end_date)
            
            task_start_date = max(dep_end_dates) if dep_end_dates else current_date
            
            # Assign to agent
            assigned_agent = task.get("assigned_agent") or self._assign_agent(
                task, agent_capacity, agent_schedule, task_start_date
            )
            
            # Calculate duration
            effort_hours = task.get("estimated_effort_hours", task.get("effort_hours", 8.0))
            capacity = agent_capacity.get(assigned_agent, self.work_hours_per_day)
            duration_days = max(1.0, effort_hours / capacity)
            
            # Calculate end date
            task_end_date = task_start_date + timedelta(days=int(duration_days))
            
            # Update agent schedule
            agent_schedule[assigned_agent].append({
                "start": task_start_date,
                "end": task_end_date,
                "task_id": task_id,
            })
            
            scheduled_task = {
                "task_id": task_id,
                "task_name": task.get("task_name", "Unnamed Task"),
                "start_date": task_start_date.isoformat(),
                "end_date": task_end_date.isoformat(),
                "duration_days": round(duration_days, 1),
                "assigned_agent": assigned_agent,
                "status": task.get("status", "planned"),
                "dependencies": task.get("dependencies", []),
                "effort_hours": round(effort_hours, 1),
                "completion_percentage": task.get("completion_percentage", 0.0),
            }
            
            scheduled.append(scheduled_task)
            
            # Update queue with dependent tasks
            for dependent in dependency_graph[task_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return scheduled
    
    def _assign_agent(
        self,
        task: Dict[str, Any],
        agent_capacity: Dict[str, float],
        agent_schedule: Dict[str, List[Dict[str, Any]]],
        start_date: datetime,
    ) -> str:
        """Assign task to available agent."""
        # Simple assignment: find agent with least workload
        if not agent_capacity:
            return "unassigned"
        
        # Calculate current workload for each agent
        agent_workload = {}
        for agent, capacity in agent_capacity.items():
            current_work = sum(
                1 for schedule in agent_schedule.get(agent, [])
                if schedule["end"] > start_date
            )
            agent_workload[agent] = current_work
        
        # Assign to agent with least workload
        return min(agent_workload.items(), key=lambda x: x[1])[0]
    
    def _calculate_critical_path(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Calculate critical path using longest path algorithm."""
        task_map = {task["task_id"]: task for task in tasks}
        
        # Build dependency graph
        dependency_graph = defaultdict(list)
        for task in tasks:
            for dep in task.get("dependencies", []):
                dependency_graph[dep].append(task["task_id"])
        
        # Calculate longest path from each node
        longest_paths = {}
        
        def dfs(task_id: str) -> float:
            if task_id in longest_paths:
                return longest_paths[task_id]
            
            task = task_map[task_id]
            max_dep_duration = 0.0
            
            for dep in task.get("dependencies", []):
                dep_duration = dfs(dep)
                max_dep_duration = max(max_dep_duration, dep_duration)
            
            duration = task.get("duration_days", 0.0)
            longest_paths[task_id] = max_dep_duration + duration
            
            return longest_paths[task_id]
        
        # Calculate longest paths
        for task in tasks:
            dfs(task["task_id"])
        
        # Find critical path (tasks on longest path)
        max_path_length = max(longest_paths.values()) if longest_paths else 0
        
        critical_path = []
        for task_id, path_length in longest_paths.items():
            if abs(path_length - max_path_length) < 0.01:  # On critical path
                critical_path.append(task_id)
        
        return critical_path
    
    def _calculate_slack(self, tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate slack (float) for each task."""
        task_map = {task["task_id"]: task for task in tasks}
        
        # Calculate earliest start/finish
        earliest_start = {}
        earliest_finish = {}
        
        def calc_earliest(task_id: str):
            if task_id in earliest_start:
                return earliest_start[task_id]
            
            task = task_map[task_id]
            max_dep_finish = 0.0
            
            for dep in task.get("dependencies", []):
                dep_finish = calc_earliest(dep)
                max_dep_finish = max(max_dep_finish, dep_finish)
            
            earliest_start[task_id] = max_dep_finish
            duration = task.get("duration_days", 0.0)
            earliest_finish[task_id] = earliest_start[task_id] + duration
            
            return earliest_finish[task_id]
        
        # Calculate latest start/finish
        project_end = max(
            (task.get("duration_days", 0.0) + calc_earliest(task["task_id"])
             for task in tasks),
            default=0.0
        )
        
        latest_finish = {}
        latest_start = {}
        
        def calc_latest(task_id: str):
            if task_id in latest_finish:
                return latest_finish[task_id]
            
            task = task_map[task_id]
            min_successor_start = project_end
            
            # Find successors
            successors = [
                t["task_id"] for t in tasks
                if task_id in t.get("dependencies", [])
            ]
            
            if successors:
                min_successor_start = min(calc_latest(succ) for succ in successors)
            
            duration = task.get("duration_days", 0.0)
            latest_finish[task_id] = min_successor_start
            latest_start[task_id] = latest_finish[task_id] - duration
            
            return latest_finish[task_id]
        
        # Calculate slack
        slack = {}
        for task in tasks:
            task_id = task["task_id"]
            calc_earliest(task_id)
            calc_latest(task_id)
            
            slack[task_id] = latest_start[task_id] - earliest_start[task_id]
        
        return slack
    
    def _identify_milestones(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify project milestones."""
        milestones = []
        
        # Group tasks by end date
        end_dates = {}
        for task in tasks:
            end_date = task.get("end_date")
            if end_date:
                # Normalize date to string for grouping
                if isinstance(end_date, datetime):
                    end_date_str = end_date.isoformat()
                elif isinstance(end_date, str):
                    end_date_str = end_date
                else:
                    continue
                
                if end_date_str not in end_dates:
                    end_dates[end_date_str] = []
                end_dates[end_date_str].append(task)
        
        # Identify significant milestones (multiple tasks completing or high priority)
        for end_date_str, tasks_on_date in end_dates.items():
            if len(tasks_on_date) >= 2 or any(
                t.get("priority", 5) >= 8 for t in tasks_on_date
            ):
                milestones.append({
                    "name": f"Milestone: {len(tasks_on_date)} tasks completed",
                    "date": end_date_str,
                    "tasks": [t["task_id"] for t in tasks_on_date],
                })
        
        # Sort by date (as string comparison works for ISO format)
        return sorted(milestones, key=lambda x: x["date"])

