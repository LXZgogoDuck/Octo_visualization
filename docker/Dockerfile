# Base Image
ARG PARENT_IMAGE
FROM ${PARENT_IMAGE}

# Define build arguments after FROM
ARG USER_ID
ARG GROUP_ID=1000  # Ensure a default GID to avoid unset errors
ARG PYTHON_VERSION=3.10

ENV NVIDIA_DRIVER_CAPABILITIES all

# Install OS-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    ffmpeg \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-utils \
    wget \
    xvfb \
    libglvnd-dev \
    && rm -rf /var/lib/apt/lists/*

# Install (mini) conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python="$PYTHON_VERSION" && \
    /opt/conda/bin/conda clean -ya

ENV PATH="/opt/conda/bin:$PATH"
SHELL ["/bin/bash", "-c"]

# NVIDIA Vulkan setup
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Install JAX with CUDA support
RUN pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy project files
COPY ./ /user/hypervla
WORKDIR /user/hypervla

# Install Python dependencies
RUN pip install -e .

# Ensure the group exists
RUN if ! getent group ${GROUP_ID}; then \
        groupadd -g ${GROUP_ID} usergroup; \
    else \
        EXISTING_GROUP=$(getent group ${GROUP_ID} | cut -d: -f1); \
        echo "Using existing group: $EXISTING_GROUP"; \
    fi

# Create user and use the correct group name
RUN useradd --shell /bin/bash -u ${USER_ID} -g $(getent group ${GROUP_ID} | cut -d: -f1) -o -d /user user && \
    chown -R user:$(getent group ${GROUP_ID} | cut -d: -f1) /user && chmod -R u+w /user

# Set Python environment variables
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER user

# Default shell
CMD ["/bin/bash"]
