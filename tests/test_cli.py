import subprocess

def test_elicipy_command():
    result = subprocess.run(["elicipy", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Usage" in result.stdout

def test_elicipy_form_command():
    result = subprocess.run(["elicipy-form", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Usage" in result.stdout
