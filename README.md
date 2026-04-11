# AI Engineer Simulation Environment

A simulation environment where an AI acts as a software engineer and is evaluated on real-world coding tasks. Fully compliant with the **OpenEnv** specification.

## Overview

This environment provides three increasingly difficult tasks that mirror real engineering work:

1. **Email Triage** (Easy) - Classify and prioritize incoming emails
2. **Code Review** (Medium) - Identify bugs and suggest improvements in Python code
3. **Data Cleaning** (Hard) - Clean and validate messy CSV datasets

Each task uses a **programmatic grader** that assigns scores between 0.0 and 1.0 based on:

- Correctness of solution
- Quality of reasoning
- Efficiency of approach
- Handling of edge cases

---

## Installation

### Prerequisites

- Python 3.9+
- pip or conda
- OpenAI API key (for baseline inference)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd ai-engineer-sim
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env and add your API keys and optional defaults
```

Example `.env` contents:

```bash
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4
HF_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Quick Start

### Running a Task

```python
from env.base import Action
from env.registry import TaskRegistry

# Create environment
env = TaskRegistry.instantiate("email_triage")

# Reset to get initial observation
obs = env.reset()
print(obs.content)
print(f"Available actions: {obs.available_actions}")

# Take an action
action = Action(
    action_type="classify_email",
    payload={"category": "urgent"},
    reasoning="Server alert indicates urgent priority"
)

# Step
result = env.step(action)
print(f"Reward: {result.reward.value}")
print(f"Next observation: {result.observation.content}")
```

### Running All Tests

```bash
pytest tests/ -v
```

### Running Baseline Inference

```bash
python scripts/baseline.py \
    --model gpt-4 \
    --task email_triage \
    --episodes 3 \
    --output results.json
```

### HTTP Client (Production Deployment)

For production deployments, use the HTTP client to connect to a containerized server:

```python
from client import AIEngineerEnv
from env.base import Action

# Connect to server
env = AIEngineerEnv(base_url="http://localhost:8000", task_id="email_triage")

# Reset and interact
result = env.reset()
print(f"Task: {result.observation.task_id}")
print(f"Available actions: {result.observation.available_actions}")

# Take actions
action = Action(
    action_type="classify_email",
    payload={"category": "urgent"},
    reasoning="Server alert indicates urgent priority"
)
result = env.step(action)
print(f"Reward: {result.reward.value}")

# Get episode state
state = env.state()
print(f"Total reward: {state.cumulative_reward}")
```

**Starting the Server**:

```bash
# Start inference server
uvicorn inference:app --host 0.0.0.0 --port 8000

# Or with Docker
docker build -t ai-engineer-sim .
docker run -p 8000:8000 ai-engineer-sim
```

---

## Task Specifications

### 1. Email Triage (Easy)

**Objective**: Classify emails into categories and assign priority levels.

**State Space**:

- Current email content (subject, body, sender)
- Available categories: `meeting`, `work`, `urgent`, `policy`, `spam`
- Priority levels: 0 (urgent), 1 (high), 2 (medium), 3 (low)

**Action Space**:

- `classify_email`: Assign a category to an email
- `prioritize_email`: Assign a priority level (0-3)
- `submit_classification`: Submit final classifications

**Grading Criteria** (per step):

- **Correctness** (+0.5 if category matches ground truth, -0.2 otherwise)
- **Reasoning Quality** (+0.1 if reasoning provided, 0.0 otherwise)
- **Priority Accuracy** (+0.5 if within ±1 of true priority)

**Example Workflow**:

```
Step 1: classify_email("meeting") → +0.6 reward (correct + reasoning)
Step 2: prioritize_email(1) → +0.5 reward
Step 3: submit_classification() → Task score = 80% accuracy
```

**Sample Data**: 8 diverse emails with varying categories and priorities

---

### 2. Code Review (Medium)

**Objective**: Identify bugs and suggest improvements in Python code.

**State Space**:

- Current code snippet
- Programming language (Python)
- Snippet difficulty level (1-3)

**Action Space**:

- `identify_bug`: Describe a bug found in the code
- `suggest_fix`: Propose a fix for an identified bug
- `submit_review`: Submit the complete code review

**Grading Criteria**:

- **Bug Detection** (+0.3 per correct bug, -0.1 per false positive)
- **Fix Quality** (+0.3 per valid fix, 0.0 for incomplete fixes)
- **Reasoning** (+0.2 for detailed technical reasoning)

**Example Workflow**:

```
Step 1: identify_bug("Division by zero when list is empty") → +0.3
Step 2: suggest_fix("Check list length before division") → +0.5
Step 3: submit_review() → Coverage score calculated
```

**Sample Data**: 5 code snippets with varying complexity

---

### 3. Data Cleaning (Hard)

**Objective**: Clean and validate a messy CSV dataset.

**State Space**:

- Current row being cleaned (name, age, email, phone, join_date, salary)
- Data quality issues (missing values, invalid formats, outliers)

**Action Space**:

- `handle_missing_value`: Fill or mark missing data
- `fix_format`: Correct data type/format issues
- `validate_field`: Check field validity
- `remove_invalid_row`: Remove rows with unfixable errors
- `submit_cleaned_data`: Submit cleaned dataset

**Grading Criteria**:

- **Format Fixes** (+0.2 per valid correction)
- **Missing Values** (+0.15 per reasonable imputation)
- **Field Validation** (+0.1 per validated field)
- **Row Removal** (+0.25 per rejected invalid row)

**Validation Rules**:

- `age`: 18-80
- `email`: Must contain @ and .
- `phone`: Format XXX-XXXX
- `join_date`: Valid date, not in future
- `salary`: Positive number

**Example Workflow**:

```
Step 1: fix_format("email": "john@example.com") → +0.2
Step 2: handle_missing_value("age": "35") → +0.15
Step 3: remove_invalid_row(5) → +0.25
Step 4: submit_cleaned_data() → Final score = 8/10 rows clean
```

**Sample Data**: CSV file with 10 rows containing realistic data quality issues

---

## OpenEnv Interface

All environments fully implement the OpenEnv specification:

### Core Methods

```python
# Reset the environment
observation: Observation = env.reset()

# Take an action
result: StepResult = env.step(action: Action)
# Returns: StepResult(observation, reward, done, info)

# Get current state snapshot
state: EnvState = env.state()
```

### Data Models

**Observation**:

```python
class Observation(BaseModel):
    task_id: str                    # Task identifier
    step: int                       # Step number in episode
    content: str                    # Human-readable state description
    context: dict[str, Any]         # Structured task data
    available_actions: list[str]    # Valid action names
    metadata: dict[str, Any]        # Additional info
```

**Action**:

```python
class Action(BaseModel):
    action_type: str                # Name of action to perform
    payload: dict[str, Any]         # Action parameters
    reasoning: str = ""             # Optional explanation
```

**Reward**:

```python
class Reward(BaseModel):
    value: float                    # Immediate step reward (-1.0 to 1.0)
    cumulative: float               # Total accumulated reward
    breakdown: dict[str, float]     # Per-criterion scores
    message: str = ""               # Explanation
```

---

## Reward Function

The reward function encourages incremental progress through:

1. **Step-wise Rewards**: Positive rewards for correct actions at each step
2. **Loop Penalty**: -0.2 penalty if same action used 3+ times consecutively
3. **Final Score**: Bonus reward on task submission based on overall accuracy

### Reward Range

- **-1.0**: Completely incorrect or destructive action
- **0.0**: No clear value, neutral action
- **+1.0**: Perfect solution or major milestone

### Example Reward Trajectory

```
Email Triage Episode:
Step 1: classify_email → +0.6 (correct + reasoning)
Step 2: classify_email → +0.5 (correct)
Step 3: classify_email → -0.2 (wrong category)
Step 4: prioritize_email → +0.3 (useful action)
Step 5: submit_classification → +0.4 (80% accuracy)
────────────────────────────────
Total: +1.8 cumulative reward
```

---

## Baseline Results

Expected baseline performance with GPT-4:

| Task          | Episodes | Mean Reward | Max Reward | Accuracy |
| ------------- | -------- | ----------- | ---------- | -------- |
| Email Triage  | 3        | 0.72        | 0.85       | 87%      |
| Code Review   | 3        | 0.58        | 0.75       | 65%      |
| Data Cleaning | 3        | 0.45        | 0.62       | 58%      |

**Running Baseline**:

```bash
python scripts/baseline.py \
    --model gpt-4 \
    --task email_triage \
    --episodes 3 \
    --output baseline_results.json
```

---

## Project Structure

```
ai-engineer-sim/
├── env/
│   ├── __init__.py
│   ├── base.py                     # Core OpenEnv classes
│   └── registry.py                 # Task registry
├── tasks/
│   ├── __init__.py
│   ├── email_triage.py             # Email classification task
│   ├── code_review.py              # Code review task
│   └── data_cleaning.py            # Data cleaning task
├── tests/
│   ├── __init__.py
│   ├── test_base.py
│   ├── test_email_triage.py
│   ├── test_code_review.py
│   └── test_data_cleaning.py
├── scripts/
│   └── baseline.py                 # Inference script
├── data/
│   ├── emails.json                 # Email dataset
│   ├── code_snippets.json          # Code snippets
│   └── dirty_data.csv              # Messy CSV data
├── openenv.yaml                    # OpenEnv metadata
├── requirements.txt                # Dependencies
├── Dockerfile                      # Container image
├── .env.example                    # Environment variables template
├── .gitignore
└── README.md                       # This file
```

---

## Docker Deployment

### Build Image

```bash
docker build -t ai-engineer-sim .
```

### Run Container

```bash
docker run \
    -e OPENAI_API_KEY="your-key-here" \
    -e HF_TOKEN="your-token-here" \
    ai-engineer-sim \
    python scripts/baseline.py --task email_triage
```

### Deploy to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Select "Docker" as the runtime
3. Push the repository to the Space
4. Configure environment variables in Space settings
5. The environment will automatically deploy

---

## Development & Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific task
pytest tests/test_email_triage.py -v

# With coverage
pytest tests/ --cov=env --cov=tasks
```

### Adding a New Task

1. Create `tasks/new_task.py` extending `OpenEnvBase`
2. Implement required methods:
   - `_build_initial_observation()`
   - `_grade(action, obs)`
   - `_apply(action, obs)`
   - `_is_done(obs, reward)`
3. Register with decorator: `@TaskRegistry.register("new_task_id")`
4. Add test file: `tests/test_new_task.py`
5. Update `tasks/__init__.py`

---

## API Reference

### Creating an Environment

```python
from env.registry import TaskRegistry

# By task ID
env = TaskRegistry.instantiate("email_triage")

# Get all registered tasks
tasks = TaskRegistry.all_tasks()  # → ['email_triage', 'code_review', 'data_cleaning']
```

### Episode Loop

```python
env = TaskRegistry.instantiate("email_triage")
obs = env.reset()

done = False
total_reward = 0

while not done:
    # Agent selects action based on observation
    action = agent.select_action(obs)

    # Environment steps
    result = env.step(action)

    # Unpack result
    obs = result.observation
    reward = result.reward
    done = result.done
    info = result.info

    total_reward += reward.value

print(f"Episode completed with reward: {total_reward}")
```

---

## Performance Tips

1. **Caching**: Reuse environment instances across episodes
2. **Async**: Use async inference for parallel episode evaluation
3. **Batching**: Process multiple actions in batch mode
4. **Logging**: Enable detailed logging for debugging

---

## Troubleshooting

### ImportError: No module named 'openai'

```bash
pip install openai
```

### OPENAI_API_KEY not found

Ensure `.env` file exists and contains:

```
OPENAI_API_KEY=sk-...
```

### Tests fail with RuntimeError

Make sure you call `env.reset()` before `env.step()`

### Docker build fails

Check that all files are in the repository and requirements.txt is correct

---

## Citation

If you use this environment in your research, please cite:

```bibtex
@misc{ai-engineer-sim,
  title={AI Engineer Simulation: OpenEnv Environment for Evaluating AI Coding Capabilities},
  author={...},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/spaces/...}}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## Support

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Documentation**: See docstrings in source files

---

## Acknowledgments

- OpenEnv specification: [Link to specs]
- Hugging Face Spaces for deployment support
- Community contributions and feedback
