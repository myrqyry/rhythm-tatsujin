import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

class ExperimentTracker:
    def __init__(self, experiment_name, config, experiments_dir="experiments"):
        self.experiment_name = experiment_name
        self.config = config
        self.experiments_dir = Path(experiments_dir)
        self.experiment_id = self._generate_experiment_id()
        self.experiment_path = self.experiments_dir / self.experiment_id
        self._setup_experiment_dir()
        self.metadata = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "git_commit_hash": self._get_git_commit_hash(),
            "config": self.config,
            "metrics": {},
            "artifacts": [],
            "end_time": None,
            "status": "running"
        }

    def _generate_experiment_id(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_hash = str(uuid.uuid4())[:8]
        return f"{timestamp}_{self.experiment_name}_{unique_hash}"

    def _get_git_commit_hash(self):
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except Exception:
            return "git not available"

    def _setup_experiment_dir(self):
        self.experiment_path.mkdir(parents=True, exist_ok=True)

    def log_metric(self, metric_name, value):
        self.metadata["metrics"][metric_name] = value

    def log_metrics(self, metrics_dict):
        self.metadata["metrics"].update(metrics_dict)

    def save_artifact(self, file_path):
        source = Path(file_path)
        if not source.is_file():
            raise FileNotFoundError(f"Artifact does not exist: {source}")
        artifact_dir = self.experiment_path / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        destination = artifact_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        self.metadata["artifacts"].append(str(destination.relative_to(self.experiment_path)))
        return destination

    def finalize(self, status="completed"):
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["status"] = status
        self._save_metadata()

    def _save_metadata(self):
        metadata_path = self.experiment_path / "experiment.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.finalize(status="failed")
        else:
            self.finalize()
