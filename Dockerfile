FROM texlive/texlive:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MATPLOTLIB_BACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    bibtool \
    build-essential \
    ca-certificates \
    curl \
    fontconfig \
    git \
    make \
    python3 \
    python3-pip \
    python3-venv \
    rsync \
    zsh \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace

RUN bash ./setup.sh \
 && bash ./reproduce/reproduce_environment_comp_uv.sh

ENV PATH="/workspace/.venv-linux-x86_64/bin:/workspace/.venv/bin:${PATH}"

CMD ["bash"]
