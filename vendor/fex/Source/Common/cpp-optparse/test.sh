#!/bin/bash
# -*- mode: sh; coding: utf-8; indent-tabs-mode: nil -*-
# vim: set filetype=sh fileencoding=utf-8 expandtab sw=4 sts=4:

export COLUMNS=80

c () {
    echo "$(printf "%q " ./testprog "$@")"
    local t_stdout_cpp=$(mktemp -t cpp-stdout-optparse.XXXXXXXXXX)
    local t_stderr_cpp=$(mktemp -t cpp-stderr-optparse.XXXXXXXXXX)
    local t_stdout_pyt=$(mktemp -t pyt-stdout-optparse.XXXXXXXXXX)
    local t_stderr_pyt=$(mktemp -t pyt-stderr-optparse.XXXXXXXXXX)
    ./testprog "$@" >"$t_stdout_cpp" 2>"$t_stderr_cpp"
    status_cpp=$?
    ./t/testprog "$@" >"$t_stdout_pyt" 2>"$t_stderr_pyt"
    status_pyt=$?
    if ! cmp -s "$t_stderr_cpp" "$t_stderr_pyt" ; then
        diff -au "$t_stderr_cpp" "$t_stderr_pyt"
        exit 1
    fi
    rm -f "$t_stderr_cpp" "$t_stderr_pyt"
    if ! cmp -s "$t_stdout_cpp" "$t_stdout_pyt" ; then
        diff -au "$t_stdout_cpp" "$t_stdout_pyt"
        exit 1
    fi
    rm -f "$t_stdout_cpp" "$t_stdout_pyt"
    if [[ $status_cpp -ne $status_pyt ]] ; then
        echo >&2 "status $status_pyt expected, got $status_cpp"
        exit 1
    fi
}

c
c --str # ambiguous option
c -Z # unknown argument
c --argument-does-not-exist
c --version
c -h
c --help
c "foo bar" baz ""
c --clear
c --no-clear
c --clear --no-clear
c --clear --no-clear --clear
c --string "foo bar"
c --string # requires argument
c -x foo
c --clause foo
c --sentence foo
c -k -k -k -k -k
c --verbose
c -s
c --silent
c -v -s
c -n-10
c -n 300
c --number=0
c -H
c -V
c -i-10
c -i 300
c --int=0
# c -i 2.3 # TODO: ignores suffix
c -i no-number
c -f-2.3
c -f 300
c --float=0
c -f no-number
c -c-2.3
c -c 300
c --complex=0
c -c no-number
c -C foo
c --choices baz
c -C wrong-choice
c --choices-list item2
c --choices-list wrong-item
c -m a -m b
c -m a --more b -m c
c --more-milk --more-milk
c --hidden foo
c -K -K -K
c --string-callback x
c --no-clear foo bar -k -k z -v -n3 "x y" -i 8 -f 3.2 -c 2
DISABLE_INTERSPERSED_ARGS=1 c -k a -k b
DISABLE_USAGE=1 c --argument-does-not-exist
DISABLE_USAGE=1 c --help
