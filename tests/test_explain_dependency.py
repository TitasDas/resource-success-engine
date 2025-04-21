# Tests the SHAP dependence script runs without error for a valid feature

import subprocess

def test_explain_dependence_script_runs():
    result = subprocess.run(
        ["python", "src/explain_dependence.py", "--feature", "cost_variance_pct"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"explain_dependence.py failed: {result.stderr}"
