"""
Data Cleaning Task - Hard
Clean inconsistent, incomplete, and invalid data in a CSV file.
"""

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from env.base import Action, Observation, OpenEnvBase, Reward
from env.registry import TaskRegistry


@TaskRegistry.register("data_cleaning")
class DataCleaningEnv(OpenEnvBase):
    """
    Hard task: Clean a messy CSV dataset with multiple issues:
    - Missing values
    - Invalid data types
    - Format inconsistencies
    - Outliers
    - Invalid emails/phones
    
    Agent must fix these issues and validate the cleaned data.
    """

    MAX_STEPS = 20
    
    VALID_ACTIONS = [
        "handle_missing_value",
        "fix_format",
        "validate_field",
        "remove_invalid_row",
        "submit_cleaned_data",
    ]

    def __init__(self, task_id: str = "data_cleaning") -> None:
        super().__init__(task_id=task_id)
        
        # Load raw data
        data_path = Path(__file__).parent.parent / "data" / "dirty_data.csv"
        with open(data_path) as f:
            self.raw_data = f.read()
        
        # Parse raw data
        reader = csv.DictReader(io.StringIO(self.raw_data))
        self.rows = list(reader)
        self.original_row_count = len(self.rows)
        
        self.cleaned_rows = []
        self.current_row_index = 0
        self.issues_fixed = {
            "missing_values": 0,
            "format_fixes": 0,
            "removed_rows": 0,
            "validated_fields": 0,
        }
        
        # Define validation rules
        self.validation_rules = {
            "name": lambda x: len(x) > 0,
            "age": lambda x: x == "" or (x.isdigit() and 18 <= int(x) <= 80),
            "email": lambda x: "@" in x and "." in x and len(x) > 5,
            "phone": lambda x: x == "" or bool(re.match(r"^\d{3}-\d{4}$", x)),
            "join_date": lambda x: x == "" or self._is_valid_date(x),
            "salary": lambda x: x == "" or (x.isdigit() and int(x) > 0),
        }

    @staticmethod
    def _is_valid_date(date_str: str) -> bool:
        """Check if date is valid and not in future."""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj <= datetime.now()
        except ValueError:
            return False

    def _build_initial_observation(self) -> Observation:
        """Present the data cleaning task."""
        preview = self.raw_data.split("\n")[:5]
        
        return Observation(
            task_id=self.task_id,
            step=0,
            content=f"Clean and validate CSV data with {self.original_row_count} rows.\n\nPreview:\n" + "\n".join(preview),
            context={
                "total_rows": self.original_row_count,
                "columns": ["name", "age", "email", "phone", "join_date", "salary"],
                "issues": [
                    "Missing values (empty cells)",
                    "Invalid email formats",
                    "Invalid phone formats",
                    "Invalid dates (future dates)",
                    "Negative salaries",
                    "Age out of range (>80)",
                ],
            },
            available_actions=self.VALID_ACTIONS,
            metadata={
                "instructions": "Fix each row systematically, removing invalid rows.",
            },
        )

    def _grade(self, action: Action, obs: Observation) -> Reward:
        """Score data cleaning actions."""
        breakdown = {}
        message = ""
        value = 0.0
        
        if action.action_type == "handle_missing_value":
            row_idx = action.payload.get("row_index", self.current_row_index)
            field = action.payload.get("field", "")
            replacement = action.payload.get("value", "")
            
            # Check if replacement is reasonable
            if field and replacement:
                is_reasonable = len(replacement) > 0
                value = 0.15 if is_reasonable else -0.1
                message = f"Missing value handled: {field}"
                self.issues_fixed["missing_values"] += 1
            else:
                value = -0.2
                message = "Invalid missing value handling"
        
        elif action.action_type == "fix_format":
            field = action.payload.get("field", "")
            fixed_value = action.payload.get("value", "")
            
            # Validate fixed format
            if field in self.validation_rules:
                is_valid = self.validation_rules[field](fixed_value)
                value = 0.2 if is_valid else -0.1
                message = f"Format fixed: {field} = {fixed_value}"
                if is_valid:
                    self.issues_fixed["format_fixes"] += 1
            else:
                value = 0.0
                message = f"Unknown field: {field}"
        
        elif action.action_type == "validate_field":
            field = action.payload.get("field", "")
            value_to_check = action.payload.get("value", "")
            
            if field in self.validation_rules:
                is_valid = self.validation_rules[field](value_to_check)
                value = 0.1 if is_valid else 0.1  # Info action, small reward
                message = f"Field validated: {field} is {'valid' if is_valid else 'INVALID'}"
                self.issues_fixed["validated_fields"] += 1
            else:
                value = 0.0
        
        elif action.action_type == "remove_invalid_row":
            row_idx = action.payload.get("row_index", self.current_row_index)
            # Removing problematic rows is good
            value = 0.25
            message = f"Invalid row removed (index {row_idx})"
            self.issues_fixed["removed_rows"] += 1
        
        elif action.action_type == "submit_cleaned_data":
            # Calculate final score
            total_fixes = sum(self.issues_fixed.values())
            
            # Award based on cleaning effort
            if total_fixes > 0:
                value = 0.3 + (total_fixes / 20) * 0.5  # Up to 0.8
            else:
                value = -0.5
            
            breakdown = self.issues_fixed.copy()
            message = f"Data cleaning submitted. Total actions: {total_fixes}"
        
        return Reward(
            value=max(-1.0, min(1.0, value)),
            cumulative=0.0,
            breakdown=breakdown,
            message=message,
        )

    def _apply(self, action: Action, obs: Observation) -> Observation:
        """Apply cleaning action and progress through data."""
        if action.action_type == "submit_cleaned_data":
            return Observation(
                task_id=self.task_id,
                step=obs.step + 1,
                content=f"Data cleaning complete. Removed {self.issues_fixed['removed_rows']} rows.",
                context={
                    "original_rows": self.original_row_count,
                    "remaining_rows": self.original_row_count - self.issues_fixed["removed_rows"],
                    "issues_fixed": self.issues_fixed,
                },
                available_actions=[],
                metadata={"done": True},
            )
        
        # Progress through rows
        self.current_row_index = min(self.current_row_index + 1, len(self.rows) - 1)
        
        if self.current_row_index < len(self.rows):
            row = self.rows[self.current_row_index]
            # Show problematic fields
            issues = []
            for field, validator in self.validation_rules.items():
                value = row.get(field, "")
                if not validator(value):
                    issues.append(f"- {field}: {value or '[EMPTY]'}")
            
            issue_text = "\n".join(issues) if issues else "No issues detected"
            
            return Observation(
                task_id=self.task_id,
                step=obs.step + 1,
                content=f"Row {self.current_row_index + 1}: {json.dumps(row)}\n\nIssues:\n{issue_text}",
                context={
                    "row_index": self.current_row_index,
                    "row_data": row,
                    "total_rows": len(self.rows),
                    "processed": self.current_row_index,
                },
                available_actions=self.VALID_ACTIONS,
            )
        else:
            return Observation(
                task_id=self.task_id,
                step=obs.step + 1,
                content="All rows processed. Submit cleaned data.",
                context={"total_rows": len(self.rows)},
                available_actions=["submit_cleaned_data"],
            )

    def _is_done(self, obs: Observation, reward: Reward) -> bool:
        """Episode ends when data is submitted."""
        return obs.metadata.get("done", False)
