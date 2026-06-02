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

if ! hash pv 2>/dev/null ; then
  echo "This script requires 'pv'"
  echo "On Ubuntu, install it with 'sudo apt-get install pv'"
  exit 1;
fi

export CXX=clang++
export LLVM_PROFILE_FILE=$(mktemp)
PROFDATA=$(mktemp)
BUILDDIR="obj/target_a64/mode_debug/symbols_on/compiler_clang++/std_c++17/simulator_aarch64/negative_testing_off/code_buffer_allocator_mmap"
RUNNER="$BUILDDIR/test/test-runner"

# Build with code coverage instrumentation enabled.
scons mode=debug coverage=on target=a64 all -j8

if [ ! -f "$RUNNER" ]; then
    echo "$RUNNER not found."
    echo "No test-runner for profiling."
    exit 1;
fi

# Count the number of tests.
tests=`$RUNNER --list | wc -l`

# Generate a raw profile for a run using all tests.
echo "Running $tests tests. This may take a while..."
$RUNNER --run-all 2>&1 | grep -P "^Running [A-Z0-9]{3,}_" | pv -lbp -w 40 -s $tests >/dev/null

# Process the raw profile data for reporting.
llvm-profdata merge -sparse $LLVM_PROFILE_FILE -o $PROFDATA

# Print a coverage report for the source files in src/
REPORT="llvm-cov report $RUNNER -instr-profile=$PROFDATA $BUILDDIR/src/"
eval $REPORT

# Log the report summary line.
eval $REPORT | tail -n1 | tr -s " " | cut -d" " -f4,7,10 | xargs -i printf "%s %s\n" `date +%s` {} >>tools/code_coverage.log

# Clean up.
rm -f $LLVM_PROFILE_FILE
rm -f $PROFDATA
