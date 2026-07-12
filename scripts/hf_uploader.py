"""Out-of-band checkpoint backup to HF Hub.

Runs detached alongside training. Reads HF_TOKEN from the environment, uploads
the latest local checkpoint immediately, then re-uploads whenever it changes
(i.e. every save_every steps). Decoupled from the training process so a bad
token / wrong repo in the trainer never blocks the backup.
"""
import os
import time

from huggingface_hub import HfApi

CKPT = "checkpoints/pretrain_a100_v4.pt"
POLL_SECS = 600

api = HfApi()
user = api.whoami()["name"]
repo = user + "/akshara-pretrain"
api.create_repo(repo, repo_type="model", exist_ok=True, private=True)
print("[uploader] repo=" + repo, flush=True)

last = None
while True:
    if os.path.exists(CKPT):
        m = os.path.getmtime(CKPT)
        if m != last:
            try:
                api.upload_file(
                    path_or_fileobj=CKPT,
                    path_in_repo="pretrain_a100_v4.pt",
                    repo_id=repo,
                    repo_type="model",
                    commit_message="auto-backup",
                )
                last = m
                mb = os.path.getsize(CKPT) // (1024 * 1024)
                print("[uploader] pushed %d MB mtime=%d" % (mb, int(m)), flush=True)
            except Exception as e:
                print("[uploader] FAILED: " + repr(e)[:200], flush=True)
    time.sleep(POLL_SECS)
