# cpp-optparse

This is yet another option parser for C++. It is modelled after the excellent
Python optparse API. Although incomplete, anyone familiar to
[optparse](http://docs.python.org/library/optparse.html) should feel at home.

- Copyright (c) 2010 Johannes Weißl
- License: MIT License

## Design decisions

- Elegant and easy usage more important than speed / flexibility
- Small size more important than feature completeness, e.g.:
  * No unicode
  * No checking for user programming errors
  * No conflict handlers
  * No adding of new actions

## FAQ

- Why not use getopt/getopt_long?
  * Not C++ / not completely POSIX
  * Too cumbersome to use, would need lot of additional code
- Why not use Boost.Program_options?
  * Boost not installed on all target platforms (esp. cluster, HPC, ...)
  * Too big to include just for option handling for many projects:
    322 *.h (44750 lines) + 7 *.cpp (2078 lines)
- Why not use tclap/Opag/Options/CmdLine/Anyoption/Argument_helper/...?
  * Similarity to Python desired for faster learning curve

## Future work

- Support nargs > 1?

## Example

```cpp
using optparse::OptionParser;

OptionParser parser = OptionParser() .description("just an example");

parser.add_option("-f", "--file") .dest("filename")
                  .help("write report to FILE") .metavar("FILE");
parser.add_option("-q", "--quiet")
                  .action("store_false") .dest("verbose") .set_default("1")
                  .help("don't print status messages to stdout");

optparse::Values options = parser.parse_args(argc, argv);
vector<string> args = parser.args();

if (options.get("verbose"))
    cout << options["filename"] << endl;
```
