# Tests that interaction plot script runs with single feature input

import subprocess

def test_explain_interactions_script_runs():
    result = subprocess.run(
        ["python", "src/explain_interactions.py", "--feature", "allocation_spikiness"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"explain_interactions.py failed: {result.stderr}"
