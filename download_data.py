from huggingface_hub import snapshot_download
import pathlib


if __name__ == "__main__":
    dataset = "v4"
    pathlib.Path("./dataset").mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="open-spaced-repetition/fsrs-dataset",
        repo_type="dataset",
        allow_patterns=[f"{dataset}/*.tsv"],
        local_dir="./dataset",
        local_dir_use_symlinks=False,
    )
    for file in pathlib.Path(f"./dataset/{dataset}").iterdir():
        if file.suffix == ".tsv":
            file.rename(f"./dataset/{file.stem}.tsv")
    pathlib.Path(f"./dataset/{dataset}").rmdir()
