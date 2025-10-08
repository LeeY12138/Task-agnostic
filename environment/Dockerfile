# hash:sha256:151a802a16dfe3718a41a3014c967c4851f44a672fde4b05404441c0e5c20f03
FROM registry.codeocean.com/codeocean/miniconda3:4.5.11-python3.7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

COPY postInstall /
RUN /postInstall
