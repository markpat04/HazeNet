# HazeNet — RunPod training image.
# Base provides CUDA + PyTorch; we add the scientific stack + the package.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

# system libs for rasterio/netCDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev gdal-bin libgeos-dev && rm -rf /var/lib/apt/lists/*

# python deps (torch already in base image)
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > req.txt && pip install --no-cache-dir -r req.txt

COPY hazenet ./hazenet
COPY configs ./configs

ENV PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE

# datacube.zarr is mounted from the persistent volume at /workspace/data.
# Override --stage / --config at run time, e.g.:
#   docker run --gpus all -v /vol:/workspace/data hazenet \
#       --config configs/runpod.yaml --stage train,eval
ENTRYPOINT ["python", "-m", "hazenet.cli"]
CMD ["--config", "configs/runpod.yaml", "--stage", "train,eval"]
