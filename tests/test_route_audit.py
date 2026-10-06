import subprocess
import sys
from pathlib import Path


def test_route_audit_inspects_live_routes():
    result = subprocess.run(
        [sys.executable, "tools/route_audit.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    content = Path("docs/flask-migration-status.md").read_text()
    assert "FastAPI routes discovered:" in content
    assert "No stale web.server" in content
    assert "POST /api/generate-chart" in content
    assert "POST /api/upload-audio" in content
