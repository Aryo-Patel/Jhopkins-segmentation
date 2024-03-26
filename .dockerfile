FROM ubuntu:20.20

SHELL ["/bin/bash", "--login", "-c"]

# create a non-root user
ARG username=ec2-user
ARG uid=1000
ARG gid=100

ENV USER=${username} UID=${uid} GID=${gid} HOME=/home/${username}

# copy the config files in (this is done as a root user)
COPY environment.yml /tmp/