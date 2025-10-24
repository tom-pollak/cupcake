# cupcake

cd torchstore && uv pip install .

cd vllm
python use_existing_torch.py
uv pip install setuptools-scm
uv pip install . --no-build-isolation --prerelease=allow -v
