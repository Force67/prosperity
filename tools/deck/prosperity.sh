#!/bin/sh
# Steam Deck launcher. A Game Mode launch has no console, so the log goes to
# last.log beside the executable.
cd "$(dirname "$0")" && exec ./ps4delta "$@" > last.log 2>&1
