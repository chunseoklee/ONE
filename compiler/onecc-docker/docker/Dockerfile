# Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

FROM ubuntu:20.04

ARG VERSION

RUN apt-get update && apt-get install -qqy --no-install-recommends \
    wget \
    ca-certificates \
    && wget --no-check-certificate https://github.com/Samsung/ONE/releases/download/${VERSION}/one-compiler-focal_${VERSION}_amd64.deb \
    && apt-get install -y ./one-compiler-focal_${VERSION}_amd64.deb \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["onecc"]
