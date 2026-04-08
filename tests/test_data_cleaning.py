"""
Unit tests for DataCleaningEnv task.
"""

import pytest

from env.base import Action
from tasks.data_cleaning import DataCleaningEnv


@pytest.fixture
def env():
    """Create a DataCleaningEnv instance."""
    return DataCleaningEnv()


def test_reset_returns_observation(env):
    """Test that reset() returns valid initial observation."""
    obs = env.reset()
    
    assert obs.task_id == "data_cleaning"
    assert obs.step == 0
    assert "Clean and validate CSV" in obs.content
    assert "columns" in obs.context
    assert len(obs.available_actions) > 0


def test_handle_missing_value(env):
    """Test handling a missing value."""
    env.reset()
    
    action = Action(
        action_type="handle_missing_value",
        payload={
            "row_index": 0,
            "field": "age",
            "value": "35",
        },
        reasoning="Missing age, estimated from context",
    )
    
    result = env.step(action)
    assert result.reward.value >= 0
    assert result.observation.step == 1


def test_fix_format(env):
    """Test fixing a data format."""
    env.reset()
    
    action = Action(
        action_type="fix_format",
        payload={
            "field": "email",
            "value": "john@example.com",
        },
        reasoning="Fixed malformed email",
    )
    
    result = env.step(action)
    assert result.reward.value > 0  # Valid email format


def test_fix_invalid_format(env):
    """Test penalizing fixes that don't improve format."""
    env.reset()
    
    action = Action(
        action_type="fix_format",
        payload={
            "field": "email",
            "value": "notanemail",  # Invalid email
        },
        reasoning="Attempted fix",
    )
    
    result = env.step(action)
    assert result.reward.value <= 0  # Invalid format should be penalized


def test_validate_field(env):
    """Test validating a field."""
    env.reset()
    
    action = Action(
        action_type="validate_field",
        payload={
            "field": "age",
            "value": "35",
        },
    )
    
    result = env.step(action)
    assert result.reward.value >= 0
    assert "valid" in result.reward.message.lower()


def test_remove_invalid_row(env):
    """Test removing an invalid row."""
    env.reset()
    
    action = Action(
        action_type="remove_invalid_row",
        payload={"row_index": 0},
        reasoning="Row contains too many errors",
    )
    
    result = env.step(action)
    assert result.reward.value > 0  # Removing bad data should be rewarded


def test_submit_cleaned_data(env):
    """Test submitting cleaned data."""
    env.reset()
    
    # Perform some cleaning actions
    actions = [
        Action(action_type="fix_format", payload={"field": "email", "value": "john@example.com"}),
        Action(action_type="validate_field", payload={"field": "age", "value": "35"}),
    ]
    
    for action in actions:
        env.step(action)
    
    # Submit cleaned data
    submit_action = Action(
        action_type="submit_cleaned_data",
        payload={},
    )
    result = env.step(submit_action)
    assert result.done is True


def test_validation_rules(env):
    """Test various validation rules."""
    env.reset()
    
    # Valid email
    assert env.validation_rules["email"]("john@example.com") is True
    
    # Invalid email
    assert env.validation_rules["email"]("notanemail") is False
    
    # Valid phone
    assert env.validation_rules["phone"]("555-1234") is True
    
    # Invalid phone
    assert env.validation_rules["phone"]("555-123") is False
    
    # Valid age
    assert env.validation_rules["age"]("25") is True
    
    # Invalid age (too old)
    assert env.validation_rules["age"]("150") is False


def test_date_validation(env):
    """Test date validation."""
    env.reset()
    
    # Valid past date
    assert env.validation_rules["join_date"]("2020-01-15") is True
    
    # Future date (should fail)
    assert env.validation_rules["join_date"]("2030-12-31") is False
    
    # Invalid date format
    assert env.validation_rules["join_date"]("01/15/2020") is False


def test_step_before_reset(env):
    """Test that step() before reset() raises error."""
    action = Action(
        action_type="handle_missing_value",
        payload={"row_index": 0, "field": "age", "value": "35"},
    )
    
    with pytest.raises(RuntimeError):
        env.step(action)


def test_invalid_action_type(env):
    """Test that invalid action types raise error."""
    env.reset()
    
    action = Action(
        action_type="invalid_cleaning_action",
        payload={},
    )
    
    with pytest.raises(ValueError):
        env.step(action)
