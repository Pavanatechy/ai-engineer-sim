"""
Code Review Task - Medium
Identify bugs and suggest improvements in Python code snippets.
"""

import json
from pathlib import Path
from typing import Any

from env.base import Action, Observation, OpenEnvBase, Reward
from env.registry import TaskRegistry


@TaskRegistry.register("code_review")
class CodeReviewEnv(OpenEnvBase):
    """
    Medium task: Review Python code snippets for bugs and improvements.
    
    Grading criteria:
    - Correctly identify all bugs
    - Suggest meaningful fixes
    - Provide quality reasoning
    """

    MAX_STEPS = 20
    
    VALID_ACTIONS = ["identify_bug", "suggest_fix", "submit_review"]

    def __init__(self, task_id: str = "code_review") -> None:
        super().__init__(task_id=task_id)
        
        # Load code snippets from data file
        data_path = Path(__file__).parent.parent / "data" / "code_snippets.json"
        with open(data_path) as f:
            self.snippets = json.load(f)
        
        self.current_snippet_index = 0
        self.identified_bugs: dict[str, list[str]] = {}  # snippet_id -> [bugs]
        self.suggested_fixes: dict[str, str] = {}  # snippet_id -> fix
        self.bug_scores = {}

    def _build_initial_observation(self) -> Observation:
        """Return the first code snippet."""
        snippet = self.snippets[0]
        return Observation(
            task_id=self.task_id,
            step=0,
            content=f"Review code snippet {snippet['id']}:\n\n```python\n{snippet['snippet']}\n```",
            context={
                "snippet_id": snippet["id"],
                "code": snippet["snippet"],
                "difficulty": snippet["difficulty"],
                "total_snippets": len(self.snippets),
                "current_index": 0,
            },
            available_actions=self.VALID_ACTIONS,
            metadata={
                "instructions": "Identify bugs, suggest fixes, then submit your review.",
            },
        )

    def _grade(self, action: Action, obs: Observation) -> Reward:
        """Score bug identification and fix suggestions."""
        snippet_id = obs.context.get("snippet_id")
        snippet = next((s for s in self.snippets if s["id"] == snippet_id), None)
        if not snippet:
            return Reward(value=0.0, cumulative=0.0)
        
        breakdown = {}
        value = 0.0
        message = ""
        
        if action.action_type == "identify_bug":
            bug = action.payload.get("bug", "")
            expected_bugs = snippet.get("bugs", [])
            
            # Check if identified bug matches any expected bug (fuzzy match)
            is_valid_bug = any(
                keyword in bug.lower() 
                for keywords in expected_bugs 
                for keyword in keywords.split()
            )
            
            if is_valid_bug:
                breakdown["bug_found"] = 0.3
                value = 0.3
                message = f"Bug identified: {bug}"
            else:
                breakdown["bug_not_found"] = -0.1
                value = -0.1
                message = f"Bug '{bug}' not clearly identified in code"
            
            # Store for later
            if snippet_id not in self.identified_bugs:
                self.identified_bugs[snippet_id] = []
            self.identified_bugs[snippet_id].append(bug)
        
        elif action.action_type == "suggest_fix":
            fix = action.payload.get("fix", "")
            expected_fix = snippet.get("expected_fix", "")
            
            # Check fix quality (simple heuristic)
            fix_quality = 0.3 if len(fix) > 10 else 0.0
            breakdown["fix_quality"] = fix_quality
            
            # Reasoning bonus
            reasoning_bonus = 0.2 if action.reasoning and len(action.reasoning) > 20 else 0.0
            breakdown["reasoning"] = reasoning_bonus
            
            value = fix_quality + reasoning_bonus
            message = f"Fix suggested: {fix[:50]}..."
            
            self.suggested_fixes[snippet_id] = fix
        
        elif action.action_type == "submit_review":
            # Calculate overall score
            if self.identified_bugs and self.suggested_fixes:
                bugs_identified = len([b for bugs in self.identified_bugs.values() for b in bugs])
                fixes_provided = len(self.suggested_fixes)
                
                # Score based on coverage
                coverage = min(bugs_identified / 3, 1.0)  # Assuming ~3 bugs per snippet
                value = 0.4 * coverage + 0.2
                breakdown["coverage"] = coverage
                message = f"Review submitted: {bugs_identified} bugs, {fixes_provided} fixes"
            else:
                value = -0.3
                message = "Incomplete review"
        
        return Reward(
            value=max(-1.0, min(1.0, value)),
            cumulative=0.0,
            breakdown=breakdown,
            message=message,
        )

    def _apply(self, action: Action, obs: Observation) -> Observation:
        """Apply action and move to next snippet or end."""
        if action.action_type == "submit_review":
            return Observation(
                task_id=self.task_id,
                step=obs.step + 1,
                content="Code review completed.",
                context={
                    "total_snippets": len(self.snippets),
                    "reviewed": len(self.identified_bugs),
                },
                available_actions=[],
                metadata={"done": True},
            )
        
        # Move to next snippet if we've identified and fixed current one
        processed = obs.context.get("current_index", 0)
        self.current_snippet_index = min(processed + 1, len(self.snippets) - 1)
        
        if self.current_snippet_index < len(self.snippets):
            snippet = self.snippets[self.current_snippet_index]
            return Observation(
                task_id=self.task_id,
                step=obs.step + 1,
                content=f"Review code snippet {snippet['id']}:\n\n```python\n{snippet['snippet']}\n```",
                context={
                    "snippet_id": snippet["id"],
                    "code": snippet["snippet"],
                    "difficulty": snippet["difficulty"],
                    "total_snippets": len(self.snippets),
                    "current_index": self.current_snippet_index,
                },
                available_actions=self.VALID_ACTIONS,
            )
        else:
            return Observation(
                task_id=self.task_id,
                step=obs.step + 1,
                content="All snippets reviewed. Ready to submit.",
                context={"total_snippets": len(self.snippets)},
                available_actions=["submit_review"],
            )

    def _is_done(self, obs: Observation, reward: Reward) -> bool:
        """Episode ends when review is submitted."""
        return obs.metadata.get("done", False)
