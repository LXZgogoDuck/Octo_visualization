#### updated Dockerfile for using GPU for SimplerEnv
# --------------------------
# Base Image
# --------------------------
ARG PARENT_IMAGE
FROM ${PARENT_IMAGE}

# --------------------------
# Arguments (set after FROM)
# --------------------------
ARG USER_ID
ARG GROUP_ID
ARG PYTHON_VERSION=3.10

# --------------------------
# Environment and Shell
# --------------------------
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "-c"]

# --------------------------
# OS-level Dependencies
# --------------------------
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
    libpng-dev \
    libvulkan1 \
    ffmpeg \
    rsync \
    tmux \
    sudo \
    unzip \
    vim \
    vulkan-tools \
    wget \
    xvfb \
    libglvnd-dev \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# Install Miniconda & Python
# --------------------------
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda init && \
    conda config --add channels https://repo.anaconda.com/pkgs/main && \
    conda config --add channels https://repo.anaconda.com/pkgs/r && \
    conda config --set show_channel_urls yes && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda install -y python="$PYTHON_VERSION" && \
    conda clean -ya

# --------------------------
# Install Python Dependencies
# --------------------------
RUN pip install --upgrade pip && \
    pip install "numpy<2" && \
    pip install git+https://github.com/google/flax.git

# --------------------------
# Install SimplerEnv
# --------------------------
WORKDIR /
RUN git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
WORKDIR /SimplerEnv
COPY simpler_requirements.txt .
RUN pip install tensorflow==2.15.0
RUN pip install tensorflow[and-cuda]==2.15.1
RUN pip install git+https://github.com/nathanrooy/simulated-annealing
RUN pip install -r simpler_requirements.txt
RUN pip install -e ./ManiSkill2_real2sim
RUN pip install -e .

# --------------------------
# Vulkan Compatibility Files
# --------------------------
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY docker/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# --------------------------
# Install Google Cloud SDK (fixed key issue)
# --------------------------
RUN apt-get update && apt-get install -y curl apt-transport-https ca-certificates gnupg
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get update && apt-get install -y google-cloud-sdk

# --------------------------
# Install RLDS Dataset Converter
# --------------------------
WORKDIR /
RUN git clone https://github.com/kpertsch/rlds_dataset_builder.git
WORKDIR rlds_dataset_builder
RUN conda env create -f environment_ubuntu.yml

# --------------------------
# Install Octo
# --------------------------
WORKDIR /user/octo
COPY ./ /user/octo
RUN pip install -e .

# --------------------------
# Install LIBERO
# --------------------------
WORKDIR /
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
WORKDIR /LIBERO
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install robosuite==1.4.1 bddl easydict
RUN pip install -e .

# --------------------------
# User and Permissions
# --------------------------
RUN groupadd -g ${GROUP_ID} usergroup || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash duser && \
    echo "duser:duser" | chpasswd && \
    adduser duser sudo && \
    chown -R ${USER_ID}:${GROUP_ID} /user && chmod -R u+w /user

# --------------------------
# Switch to User
# --------------------------
USER duser
WORKDIR /user/octo

# --------------------------
# Default CMD
# --------------------------
CMD ["/bin/bash"]
