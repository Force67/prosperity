#include "OptionParser.h"

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <complex>
#include <algorithm>

using namespace std;

using namespace optparse;

class Output {
public:
  Output(stringstream& ss, const string& d) : stream(ss), delim(d), first(true) {}
  void operator() (const string& s) {
    if (first)
      first = false;
    else
      stream << delim;
    stream << s;
  }
  stringstream& stream;
  const string& delim;
  bool first;
};

class MyCallback : public optparse::Callback {
public:
  MyCallback() : counter(0) {}
  void operator() (const Option& option, const string& opt, const string& val, const OptionParser& parser) {
    counter++;
    cout << "--- MyCallback --- " << counter << ". time called" << endl;
    cout << "--- MyCallback --- option.action(): " << option.action() << endl;
    cout << "--- MyCallback --- option.type(): " << option.type() << endl;
    cout << "--- MyCallback --- opt: " << opt << endl;
    cout << "--- MyCallback --- val: " << val << endl;
    cout << "--- MyCallback --- parser.usage(): " << parser.usage() << endl;
    cout << endl;
  }
  int counter;
};

int main(int argc, char *argv[])
{
  const string usage =
    (!getenv("DISABLE_USAGE")) ?
    "usage: %prog [OPTION]... DIR [FILE]..." : SUPPRESS_USAGE;
  const string version = "%prog 1.0\nCopyright (C) 2010 Johannes Weißl\n"
    "License GPLv3+: GNU GPL version 3 or later "
    "<http://gnu.org/licenses/gpl.html>.\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.";
  const string desc = "Lorem ipsum dolor sit amet, consectetur adipisicing"
    " elit, sed do eiusmod tempor incididunt ut labore et dolore magna"
    " aliqua.\nUt enim ad minim veniam, quis nostrud exercitation ullamco"
    " laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor"
    " in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla"
    " pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa"
    " qui officia deserunt mollit anim id est laborum.";
  const string epilog = "Sed ut perspiciatis unde omnis iste natus error sit"
    " voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque"
    " ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae"
    " dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit"
    " aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos"
    " qui ratione voluptatem sequi nesciunt.\nNeque porro quisquam est, qui"
    " dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia"
    " non numquam eius modi tempora incidunt ut labore et dolore magnam"
    " aliquam quaerat voluptatem.";

  OptionParser parser = OptionParser()
    .usage(usage)
    .version(version)
    .description(desc)
    .epilog(epilog)
  ;
  if (getenv("DISABLE_INTERSPERSED_ARGS"))
    parser.disable_interspersed_args();

  parser.set_defaults("verbosity", "50");
  parser.set_defaults("no_clear", "0");

  // test all actions
  parser.add_option("--clear") .action("store_false") .dest("no_clear") .help("clear (default)");
  parser.add_option("--no-clear") .action("store_true") .help("not clear");
  parser.add_option("--string")
    .help("This is a really long text... very long indeed! It must be wrapped on normal terminals.");
  parser.add_option("-x", "--clause", "--sentence") .metavar("SENTENCE") .set_default("I'm a sentence")
    .help("This is a really long text... very long indeed! It must be wrapped on normal terminals. "
          "Also it should appear not on the same line as the option.");
  parser.add_option("-k") .action("count") .help("how many times?");
  parser.add_option("--verbose") .action("store_const") .set_const("100") .dest("verbosity") .help("be verbose!");
  parser.add_option("-s", "--silent") .action("store_const") .set_const("0") .dest("verbosity") .help("be silent!");
  parser.add_option("-n", "--number") .type("int") .set_default("1") .metavar("NUM") .help("number of files (default: %default)");
  parser.add_option("-H") .action("help") .help("alternative help");
  parser.add_option("-V") .action("version") .help("alternative version");
  parser.add_option("-i", "--int") .action("store") .type("int") .set_default(3) .help("default: %default");
  parser.add_option("-f", "--float") .action("store") .type("float") .set_default(5.3) .help("default: %default");
  parser.add_option("-c", "--complex") .action("store") .type("complex");
  char const* const choices[] = { "foo", "bar", "baz" };
  parser.add_option("-C", "--choices") .choices(&choices[0], &choices[3]);
#if __cplusplus >= 201103L
  parser.add_option("--choices-list") .choices({"item1", "item2", "item3"});
#else
  char const* const choices_list[] = { "item1", "item2", "item3" };
  parser.add_option("--choices-list") .choices(&choices_list[0], &choices_list[3]);
#endif
  parser.add_option("-m", "--more") .action("append");
  parser.add_option("--more-milk") .action("append_const") .set_const("milk");
  parser.add_option("--hidden") .help(SUPPRESS_HELP);

  // test for 325cb47
  parser.add_option("--option1") .action("store") .type("int") .set_default(1);
  parser.add_option("--option2") .action("store") .type("int") .set_default("1");
  parser.set_defaults("option1", "640");
  parser.set_defaults("option2", 640); // now works

  MyCallback mc;
  parser.add_option("-K", "--callback") .action("callback") .callback(mc) .help("callback test");
  parser.add_option("--string-callback") .action("callback") .callback(mc) .type("string") .help("callback test");

  OptionGroup group1 = OptionGroup(parser, "Dangerous Options",
      "Caution: use these options at your own risk. "
      "It is believed that some of them\nbite.");
  group1.add_option("-g") .action("store_true") .help("Group option.") .set_default("0");
  parser.add_option_group(group1);

  OptionGroup group2 = OptionGroup(parser, "Size Options", "Image Size Options.");
  group2.add_option("-w", "--width") .action("store") .type("int") .set_default(640) .help("default: %default");
  group2.add_option("--height") .action("store") .type("int") .help("default: %default");
  parser.set_defaults("height", 480);
  parser.add_option_group(group2);

  try {
    Values& options = parser.parse_args(argc, argv);
    vector<string> args = parser.args();

    cout << "clear: " << (options.get("no_clear") ? "false" : "true") << endl;
    cout << "string: " << options["string"] << endl;
    cout << "clause: " << options["clause"] << endl;
    cout << "k: " << options["k"] << endl;
    cout << "verbosity: " << options["verbosity"] << endl;
    cout << "number: " << (int) options.get("number") << endl;
    cout << "int: " << (int) options.get("int") << endl;
    cout << "float: " << (float) options.get("float") << endl;
    complex<double> c = 0;
    if (options.is_set("complex")) {
      stringstream ss;
      ss << options["complex"];
      ss >> c;
    }
    cout << "complex: " << c << endl;
    cout << "choices: " << (const char*) options.get("choices") << endl;
    cout << "choices-list: " << (const char*) options.get("choices_list") << endl;
    {
      stringstream ss;
      for_each(options.all("more").begin(), options.all("more").end(), Output(ss, ", "));
      cout << "more: " << ss.str() << endl;
    }
    cout << "more_milk:" << endl;
    for (Values::iterator it = options.all("more_milk").begin(); it != options.all("more_milk").end(); ++it)
      cout << "- " << *it << endl;
    cout << "hidden: " << options["hidden"] << endl;
    cout << "group: " << (options.get("g") ? "true" : "false") << endl;

    cout << "option1: " << (int) options.get("option1") << std::endl;
    cout << "option2: " << (int) options.get("option2") << std::endl;

    cout << "width: " << (int) options.get("width") << std::endl;
    cout << "height: " << (int) options.get("height") << std::endl;

    cout << endl << "leftover arguments: " << endl;
    for (vector<string>::const_iterator it = args.begin(); it != args.end(); ++it) {
      cout << "arg: " << *it << endl;
    }
  }
  catch(int ex) {
    return ex;
  }

  return 0;
}
