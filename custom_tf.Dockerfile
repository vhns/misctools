FROM docker.io/tensorflow/tensorflow:2.18.0-gpu@sha256:1f16fbd9be8bb84891de12533e332bbd500511caeb5cf4db501dbe39d422f9c7

RUN apt update \
    && apt install -y --no-install-recommends libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --root-user-action ignore scikit-learn
RUN pip install --root-user-action ignore seaborn
RUN pip install --root-user-action ignore matplotlib
