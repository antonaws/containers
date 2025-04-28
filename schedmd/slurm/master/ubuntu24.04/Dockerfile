# Start from the PyTorch SageMaker base image
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-ec2-v1.34 

# Set shell and make apt-get quieter
SHELL ["bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# Define Slurm version
ARG SLURM_VERSION=master
ENV SLURM_VERSION=${SLURM_VERSION}

# Define Slurm user information
ARG SLURM_USER=slurm
ARG SLURM_USER_UID=401
ARG SLURM_USER_GID=401

# Set working directory
WORKDIR /tmp

# Create Slurm user and group
RUN set -xeuo pipefail && \
    groupadd --system --gid=${SLURM_USER_GID} ${SLURM_USER} && \
    useradd --system --no-log-init --uid=${SLURM_USER_UID} --gid=${SLURM_USER_GID} --shell=/usr/sbin/nologin ${SLURM_USER}

# Install dependencies needed for building and running Slurm
RUN set -xeuo pipefail && \
    apt-get -qq update && \
    apt-get -qq -y upgrade && \
    apt-get -qq -y install --no-install-recommends \
      build-essential fakeroot devscripts equivs curl \
      rsync gettext-base iputils-ping supervisor tini \
      apt-transport-https ca-certificates gnupg munge libmunge-dev

# Create directories for patches
RUN mkdir -p /tmp/patches

# Build Slurm packages (similar to the build stage in the original Dockerfile)
RUN set -xeuo pipefail && \
    # Download and Build Slurm
    SLURM_DIR="slurm-${SLURM_VERSION}" && \
    mkdir -p $SLURM_DIR && \
    if [ $(curl -s -Iw '%{http_code}' -o /dev/null https://download.schedmd.com/slurm/${SLURM_DIR}.tar.bz2) -eq 200 ]; then \
      curl -s -O https://download.schedmd.com/slurm/${SLURM_DIR}.tar.bz2 && \
      tar --strip-components=1 -jxvf ${SLURM_DIR}.tar.bz2 -C $SLURM_DIR; \
    else \
      curl -s -L -H "Accept: application/vnd.github+json" -O https://github.com/SchedMD/slurm/archive/${SLURM_VERSION}.tar.gz && \
      tar --strip-components=1 -zxvf ${SLURM_VERSION}.tar.gz -C $SLURM_DIR; \
    fi && \
    # Apply patches if any exist
    find $(pwd)/patches/ -type f -name "*.patch" -print0 | sort -z | xargs -t0r -n1 patch -p1 -d $SLURM_DIR -i && \
    # Build Slurm packages
    cd $SLURM_DIR && \
    mk-build-deps -ir --tool='apt-get -qq -y -o Debug::pkgProblemResolver=yes --no-install-recommends' debian/control && \
    debuild -b -uc -us >/dev/null

# Install kubectl for Kubernetes integration (same as in original Dockerfile)
ENV KUBECTL_VERSION=1.32
RUN set -xeuo pipefail && \
    apt-get -qq update && \
    apt-get -qq -y install --no-install-recommends apt-transport-https ca-certificates curl gnupg && \
    mkdir -p -m 755 /etc/apt/keyrings && \
    curl -fsSL https://pkgs.k8s.io/core:/stable:/v${KUBECTL_VERSION}/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg && \
    chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v${KUBECTL_VERSION}/deb/ /" | tee /etc/apt/sources.list.d/kubernetes.list && \
    chmod 644 /etc/apt/sources.list.d/kubernetes.list && \
    apt-get -qq update && \
    apt-get -qq -y install --no-install-recommends kubectl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Slurm packages and configure as in the slurmd target
RUN set -xeuo pipefail && \
    apt-get -qq update && \
    apt-get -qq -y install --no-install-recommends --fix-broken \
      ./slurm-smd-client_[0-9]*.deb \
      ./slurm-smd-dev_[0-9]*.deb \
      ./slurm-smd-doc_[0-9]*.deb \
      ./slurm-smd-libnss-slurm_[0-9]*.deb \
      ./slurm-smd-libpam-slurm-adopt_[0-9]*.deb \
      ./slurm-smd-libpmi2-0_[0-9]*.deb \
      ./slurm-smd-slurmd_[0-9]*.deb \
      ./slurm-smd_[0-9]*.deb && \
    rm -f *.deb && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    # Configure nsswitch.conf for Slurm integration
    cp -v /etc/nsswitch.conf{,.bak} && \
    sed -i -E "s/^passwd:[[:space:]]+/&slurm /g" /etc/nsswitch.conf && \
    sed -i -E "s/^group:[[:space:]]+/&slurm /g" /etc/nsswitch.conf

# Create necessary directories for Slurm
RUN mkdir -p /etc/slurm /var/spool/slurmd /var/log/slurm && \
    chown -R ${SLURM_USER}:${SLURM_USER} /var/spool/slurmd /var/log/slurm

# Create entrypoint script for the slurmd container
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# If a custom slurm.conf is provided via environment variable or file mount, use it\n\
if [ -n "$SLURM_CONF_CONTENT" ]; then\n\
  echo "$SLURM_CONF_CONTENT" > /etc/slurm/slurm.conf\n\
elif [ -f /etc/slurm/slurm.conf.template ]; then\n\
  envsubst < /etc/slurm/slurm.conf.template > /etc/slurm/slurm.conf\n\
fi\n\
\n\
# Start munge authentication service\n\
if [ -x /etc/init.d/munge ]; then\n\
  service munge start\n\
fi\n\
\n\
# Run slurmd in foreground\n\
exec slurmd -D -f /etc/slurm/slurm.conf\n' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

# Create a basic slurm.conf template with GPU support
RUN echo "# Basic slurm.conf template - replace at runtime with actual cluster config\n\
ClusterName=pytorch-slurm\n\
SlurmctldHost=slurmctld\n\
SlurmUser=${SLURM_USER}\n\
SlurmdSpoolDir=/var/spool/slurmd\n\
AuthType=auth/munge\n\
ProctrackType=proctrack/cgroup\n\
TaskPlugin=task/affinity,task/cgroup\n\
SchedulerType=sched/backfill\n\
SelectType=select/cons_tres\n\
SelectTypeParameters=CR_CPU_Memory\n\
GresTypes=gpu\n\
AccountingStorageType=accounting_storage/none\n\
JobAcctGatherType=jobacct_gather/none\n\
SlurmdPidFile=/var/run/slurmd.pid\n\
# Auto-detected at runtime\n\
NodeName=pytorch-node CPUs=1 State=UNKNOWN\n\
PartitionName=default Default=YES MaxTime=INFINITE State=UP\n" > /etc/slurm/slurm.conf.template

# Create a sample gres.conf file for GPU detection
RUN echo "# GPU configuration\n\
AutoDetect=nvml" > /etc/slurm/gres.conf

# OCI Annotations
LABEL \
  org.opencontainers.image.title="PyTorch with Slurm Worker Agent" \
  org.opencontainers.image.description="slurmd - The compute node daemon for Slurm integrated with PyTorch SageMaker" \
  org.opencontainers.image.documentation="https://slurm.schedmd.com/slurmd.html" \
  org.opencontainers.image.version="${SLURM_VERSION}"

# Add Red Hat OpenShift Software Certification labels
LABEL \
  name="PyTorch with Slurm Worker Agent" \
  summary="slurmd - The compute node daemon for Slurm integrated with PyTorch" \
  description="slurmd - The compute node daemon for Slurm integrated with PyTorch"

# Expose slurmd port
EXPOSE 6818/tcp

# Set entrypoint
ENTRYPOINT ["entrypoint.sh"]