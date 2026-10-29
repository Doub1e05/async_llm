"""Download the two public Qwen checkpoints into the selected user cache."""
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root=Path(__file__).resolve().parents[2]
cache=Path(os.environ.get("ROBOT_HF_HOME",os.environ.get("HF_HOME",str(Path.home()/".cache/huggingface"))))
models={}
for name in ("Qwen/Qwen3.5-0.8B","Qwen/Qwen3.5-9B"):
    path=snapshot_download(name,cache_dir=str(cache/"hub"),max_workers=4)
    models[name]=path
    print(name,path,flush=True)
output=root/"robot_sim/.runtime/models.json"
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(models,indent=2))
