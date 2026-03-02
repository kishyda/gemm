.PHONY: setup-kernels build-kernels run

setup-kernels:
	cmake -S src/CudaKernels -B src/CudaKernels/build 
build-kernels:
	cmake --build src/CudaKernels/build -- --no-print-directory

# generate-kernel-docs:
# 	generate_docs(input) {
# 		python -m nanobind.stubgen -m 
# 	}
# 	find src/CudaKernels/build -type f \( -name "*.cu") -exec generate_docs({})

run: setup-kernels build-kernels
	uv run src/main.py 