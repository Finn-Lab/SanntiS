#!/usr/bin/env python3

# Copyright 2026 EMBL - European Bioinformatics Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from sanntis.utils import http_request


def test_http_request_accepts_only_http():
    assert http_request("https://example.org/a.tar.gz").full_url == "https://example.org/a.tar.gz"
    for url in ("file:///etc/passwd", "ftp://ftp.ncbi.nlm.nih.gov/x", "/local/path"):
        with pytest.raises(ValueError, match="http"):
            http_request(url)
