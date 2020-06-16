FROM ros:melodic

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y \
      python-catkin-tools \
      python-numba \
      python-pytest \
      python-skimage \
      python-pip \
      ros-melodic-rviz \
      ros-melodic-tf2-ros \
      ros-melodic-map-server

RUN pip2 install jax==0.1.43 jaxlib==0.1.23

COPY usc_entrypoint.sh /

ENTRYPOINT ["/usc_entrypoint.sh"]

CMD ["bash"]
