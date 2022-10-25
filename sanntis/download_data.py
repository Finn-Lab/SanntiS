#!/usr/bin/env python3

# Copyright 2021 EMBL - European Bioinformatics Institute
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

import glob
import hashlib
import os
import sys
import tarfile

import requests

PKG_DIRECTORY = os.path.abspath( os.path.dirname(__file__) )
BIN_DIRECTORY = os.path.abspath( os.path.dirname(sys.argv[0]) )
INTERPROSCAN_URL = "http://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.52-86.0/interproscan-5.52-86.0-64-bit.tar.gz"
INTERPROSCAN_MD5_URL = "http://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.52-86.0/interproscan-5.52-86.0-64-bit.tar.gz.md5"

INTERPROSCAN_TAR_DEST = os.path.join(PKG_DIRECTORY, os.path.basename(INTERPROSCAN_URL))
INTERPROSCAN_DEST = os.path.join(PKG_DIRECTORY, "interproscan-5.52-86.0")


def url_file(url, dest=None, chunk_size=254):
    """ DOWNLOAD FILES FROM URL """
    try:
        r = requests.get(url, stream=True)
        if dest:
            with open(dest, "wb") as h:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    h.write(chunk)
        else:
            return r.content.decode("utf8")
    except:
        raise Exception("Download error")


def download_tar(chunk_size=254):
    """" get pkg model file from git_url """
    if os.path.isfile(INTERPROSCAN_TAR_DEST):
        print("{} exists".format(INTERPROSCAN_TAR_DEST))
        return INTERPROSCAN_TAR_DEST
    elif os.path.isdir(INTERPROSCAN_DEST):
        print("InterProScan already decompressed and in place")
        sys.exit()
    else:
        url_file(INTERPROSCAN_URL, INTERPROSCAN_TAR_DEST)
        print("Check MD5")
        #ori_md5 = url_file(INTERPROSCAN_MD5_URL).split()[0]
        ori_md5 = INTERPROSCAN_MD5
        dw_md5 = checkMD5(INTERPROSCAN_TAR_DEST)
        assert dw_md5 == ori_md5, "{} is corrupt (wrong md5 checksum)".format(
            INTERPROSCAN_TAR_DEST
        )
        print("Downloaded and correct MD5sum")


def checkMD5(file):
    """ assert MD5 of tar.gz"""
    md5_hash = hashlib.md5()
    with open(file, "rb") as h:
        models_tar = h.read()
        md5_hash.update(models_tar)
        digest = md5_hash.hexdigest()
        return digest


def decompress_file(ori, dest):
    """ decompress the models file """
    try:
        with tarfile.open(ori) as tar:
            tar.extractall(path=dest)
    except:
        raise Exception("Error decompressing")


def clean():
    """ clean tar.gz"""
    os.remove(INTERPROSCAN_TAR_DEST)


def main():
    """"""
    print("Downloading InterProScan")
    download_tar()
    print("Decompressing files")
    decompress_file(INTERPROSCAN_TAR_DEST, PKG_DIRECTORY)
    print("Cleaning...")
    clean()
    print("create symlink")
#    os.symlink( os.path.join( INTERPROSCAN_DEST, 'interproscan.sh' ), BIN_DIRECTORY )
    print("DONE!")


if __name__ == "__main__":
    main()
