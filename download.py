from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="eigenvivek/xvr-data",
    repo_type="dataset",
    local_dir="./xvr-data"
)