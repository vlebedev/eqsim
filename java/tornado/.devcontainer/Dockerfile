FROM beehivelab/tornadovm-nvidia-openjdk:latest

ENV SDKMAN_DIR /root/.sdkman
ENV MAVEN_VERSION 3.9.7
ENV JAVA_VERSION 21.0.3-ms

SHELL ["/bin/bash", "-c"]

RUN apt-get update &&\
    apt-get install -y zip unzip curl

# RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN curl -s "https://get.sdkman.io" | bash
RUN chmod a+x "$SDKMAN_DIR/bin/sdkman-init.sh"
RUN ls -la / && echo sdfjskdflsjdf

RUN set -x \
    && echo "sdkman_auto_answer=true" > $SDKMAN_DIR/etc/config \
    && echo "sdkman_auto_selfupdate=false" >> $SDKMAN_DIR/etc/config \
    && echo "sdkman_insecure_ssl=false" >> $SDKMAN_DIR/etc/config

WORKDIR $SDKMAN_DIR
RUN [[ -s "$SDKMAN_DIR/bin/sdkman-init.sh" ]] && source "$SDKMAN_DIR/bin/sdkman-init.sh" && exec "$@"

RUN source /root/.bashrc
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install java $JAVA_VERSION
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install maven $MAVEN_VERSION

WORKDIR /root

