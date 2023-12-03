import os
import sys
from pathlib import Path
from hydra import initialize, compose

sys.path.append(str(Path.cwd().parent / 'src'))
os.environ['PROJECT_ROOT'] = str(Path.cwd().parent)

with initialize(version_base="1.3", config_path="../configs"):
    cfg = compose(config_name="train.yaml")