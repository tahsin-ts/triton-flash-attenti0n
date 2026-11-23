.PHONY: test test-triton bench-cpu bench-gpu iomodel
test:            ## algorithm tests (NumPy only)
	python -m pytest -q tests/test_reference.py
test-triton:     ## Triton kernel tests (GPU, or CPU interpreter when no GPU is present)
	python -m pytest -q tests/test_triton.py -rs
bench-cpu:
	python bench/numpy_bench.py
bench-gpu:
	python bench/gpu_bench.py && python bench/gpu_bench.py --causal
iomodel:
	python flash_attn/iomodel.py
