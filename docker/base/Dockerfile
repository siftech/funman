ARG SIFT_REGISTRY_ROOT
ARG DREAL_TAG=${TARGETOS}-${TARGETARCH}
FROM ${SIFT_REGISTRY_ROOT}funman-dreal4:${DREAL_TAG}

ARG FUNMAN_BRANCH=main

# Install base dependencies
RUN apt update && apt install -y --no-install-recommends \
    curl \
    make \
    git \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install automates dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y \
    antlr4 \
    build-essential \
    gcc \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Initialize python virtual environment
ENV VIRTUAL_ENV="/funman_venv"
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ARG AUTOMATES_COMMIT_TAG
# Install automates
# RUN git clone --depth=1 https://github.com/danbryce/automates.git automates \
#     && cd automates \
#     && git fetch --depth=1 origin ${AUTOMATES_COMMIT_TAG} \
#     && git checkout ${AUTOMATES_COMMIT_TAG}
# RUN pip install -e automates

RUN pip install --no-cache-dir z3-solver==4.12.6
RUN pip install --no-cache-dir graphviz
RUN pip install --no-cache-dir pydantic
RUN pip install --no-cache-dir pydantic-settings
RUN pip install --no-cache-dir fastapi

RUN pip install --no-cache-dir --upgrade setuptools pip
RUN pip install --no-cache-dir wheel

RUN pip install /dreal4/dreal-4.21.6.2-cp38-none-manylinux_$(ldd --version | grep '^ldd' | sed -E 's/^ldd.*([0-9]+)\.([0-9]+)$/\1_\2/')_$(arch).whl

CMD [ "/bin/bash" ]
