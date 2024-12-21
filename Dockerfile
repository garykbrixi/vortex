from nvcr.io/nvidia/pytorch:24.10-py3 as base
copy requirements.txt .
run --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt
workdir /workdir
