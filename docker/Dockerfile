# Base Image
ARG PARENT_IMAGE=nvidia/cudagl:11.4.2-devel-ubuntu20.04
FROM $PARENT_IMAGE

# Arguments (User, Group, and Python Version)
ARG USER_ID=1001
ARG GROUP_ID=1001
ARG PYTHON_VERSION=3.10

# Environment Variables
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Install OS-level dependencies
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
    # Libraries for SAPIEN rendering
    libglvnd-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python="$PYTHON_VERSION" && \
    /opt/conda/bin/conda clean -ya

# Install SimplerEnv
WORKDIR /
RUN git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
WORKDIR SimplerEnv
RUN pip install --upgrade pip
RUN pip install tensorflow==2.15.0
#RUN pip install -r requirements_full_install.txt
RUN pip install git+https://github.com/nathanrooy/simulated-annealing
RUN pip install -e ./ManiSkill2_real2sim
RUN pip install -e .

# Install gsutil (Google Cloud SDK)
# Add the Google Cloud SDK key and repository securely
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee /etc/apt/sources.list.d/google-cloud-sdk.list > /dev/null

# Install Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk


# Vulkan Config for ManiSkill
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Install RLDS Dataset Converter
WORKDIR /
RUN git clone https://github.com/kpertsch/rlds_dataset_builder.git
WORKDIR rlds_dataset_builder
RUN conda env create -f environment_ubuntu.yml

# Install Octo
WORKDIR /user/octo
COPY ./ /user/octo
RUN pip install -e .
RUN pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install LIBERO
WORKDIR /
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
WORKDIR LIBERO
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install robosuite==1.4.1 bddl easydict
RUN pip install -e .

# Ensure group exists before adding user
RUN if ! getent group ${GROUP_ID}; then groupadd -g ${GROUP_ID} user; fi
# Ensure user exists before setting ownership
RUN if ! id -u ${USER_ID} >/dev/null 2>&1; then useradd --shell /bin/bash -u ${USER_ID} -g ${GROUP_ID} -m -d /user user; fi

# Fix ownership issues
RUN chown -R ${USER_ID}:${GROUP_ID} /user && chmod -R u+w /user


# Set Python Environment Variables
ENV PYTHONUNBUFFERED=1

# Default Command
CMD ["/bin/bash"]
