#!/bin/bash

# Copyright 2021, VIXL authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of ARM Limited nor the names of its contributors may be
#     used to endorse or promote products derived from this software without
#     specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This code coverage script assumes a Linux-like environment, and has been
# tested on Ubuntu 18.04.

COVERAGELOG="tools/code_coverage.log"
MONTHSECONDS=$(( 60*60*24*30 ))

if [ ! -f "$COVERAGELOG" ]; then
  echo "No code coverage log found."
  echo "Run tools/code_coverage.sh to generate one."
  exit 2;
fi

LASTCOMMIT=`git log -1 --date=format:%s | grep -P "^Date:" | grep -Po "\d+"`
LASTCOVERAGE=`tail -n1 $COVERAGELOG | cut -d' ' -f1`

d=$(( $LASTCOMMIT - $LASTCOVERAGE ))
if (( d < $MONTHSECONDS )); then
  exit 0;
fi

echo "Code coverage record too old."
echo "Run tools/code_coverage.sh to generate a newer one."
exit 1;
