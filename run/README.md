# Installation

In `open-instruct`, run:

```
conda create -n open-instruct Python==3.10
conda activate open-instruct

pip install --upgrade pip "setuptools<70.0.0" wheel
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install packaging
pip install flash-attn==2.7.2.post1 --no-build-isolation
pip install -r requirements.txt
pip install -e .
python -m nltk.downloader punkt

# extra deps
pip install ipython
cd ..
git clone https://github.com/arcee-ai/mergekit
pip install -e mergekit
```
# Running merge evals

Create merging configs in `automerging_configs` via `run/create_merge_configs.py`.

See `run/eval_automerge1.sh` for an example on running the merges.