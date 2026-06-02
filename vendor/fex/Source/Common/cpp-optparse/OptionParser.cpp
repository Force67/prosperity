/**
 * Copyright (C) 2010 Johannes Weißl <jargon@molb.org>
 * License: MIT License
 * URL: https://github.com/weisslj/cpp-optparse
 */

#include "OptionParser.h"

#include <cstdlib>
#include <algorithm>
#include <complex>
#include <ciso646>
#include <optional>

#if defined(ENABLE_NLS) && ENABLE_NLS
# include <libintl.h>
# define _(s) gettext(s)
#else
# define _(s) ((const char *) (s))
#endif

namespace optparse {

////////// auxiliary (string) functions { //////////
class str_wrap {
public:
  str_wrap(const fextl::string& l, const fextl::string& r) : lwrap(l), rwrap(r) {}
  str_wrap(const fextl::string& w) : lwrap(w), rwrap(w) {}
  fextl::string operator() (const fextl::string& s) { return lwrap + s + rwrap; }
  const fextl::string lwrap, rwrap;
};
template<typename InputIterator, typename UnaryOperator>
static fextl::string str_join_trans(const fextl::string& sep, InputIterator begin, InputIterator end, UnaryOperator op) {
  fextl::string buf;
  for (InputIterator it = begin; it != end; ++it) {
    if (it != begin)
      buf += sep;
    buf += op(*it);
  }
  return buf;
}
template<class InputIterator>
static fextl::string str_join(const fextl::string& sep, InputIterator begin, InputIterator end) {
  return str_join_trans(sep, begin, end, str_wrap(""));
}
static fextl::string& str_replace(fextl::string& s, const fextl::string& patt, const fextl::string& repl) {
  size_t pos = 0, n = patt.length();
  while (true) {
    pos = s.find(patt, pos);
    if (pos == fextl::string::npos)
      break;
    s.replace(pos, n, repl);
    pos += repl.size();
  }
  return s;
}
static fextl::string str_replace(const fextl::string& s, const fextl::string& patt, const fextl::string& repl) {
  fextl::string tmp = s;
  str_replace(tmp, patt, repl);
  return tmp;
}
static fextl::string str_format(const fextl::string& str, size_t pre, size_t len, bool running_text = true, bool indent_first = true) {
  fextl::string s = str;
  fextl::stringstream ss;
  fextl::string p;
  len -= 2; // Python seems to not use full length
  if (running_text)
    replace(s.begin(), s.end(), '\n', ' ');
  if (indent_first)
    p = fextl::string(pre, ' ');

  size_t pos = 0, linestart = 0;
  size_t line = 0;
  while (true) {
    bool wrap = false;

    size_t new_pos = s.find_first_of(" \n\t", pos);
    if (new_pos == fextl::string::npos)
      break;
    if (s[new_pos] == '\n') {
      pos = new_pos + 1;
      wrap = true;
    }
    if (line == 1)
      p = fextl::string(pre, ' ');
    if (wrap || new_pos + pre > linestart + len) {
      ss << p << s.substr(linestart, pos - linestart - 1) << std::endl;
      linestart = pos;
      line++;
    }
    pos = new_pos + 1;
  }
  ss << p << s.substr(linestart) << std::endl;
  return ss.str();
}
static fextl::string str_inc(const fextl::string& s) {
  fextl::stringstream ss;
  fextl::string v = (s != "") ? s : "0";
  long i;
  fextl::istringstream(v) >> i;
  ss << i+1;
  return ss.str();
}
static unsigned int cols() {
  unsigned int n = 80;
#ifndef _WIN32
  const char *s = getenv("COLUMNS");
  if (s)
    fextl::istringstream(s) >> n;
#endif
  return n;
}
static fextl::string basename(const fextl::string& s) {
  fextl::string b = s;
  size_t i = b.find_last_not_of('/');
  if (i == fextl::string::npos) {
    if (b[0] == '/')
      b.erase(1);
    return b;
  }
  b.erase(i+1, b.length()-i-1);
  i = b.find_last_of("/");
  if (i != fextl::string::npos)
    b.erase(0, i+1);
  return b;
}
////////// } auxiliary (string) functions //////////


////////// class OptionContainer { //////////
Option& OptionContainer::add_option(const fextl::string& opt) {
  const fextl::string tmp[1] = { opt };
  return add_option(fextl::vector<fextl::string>(&tmp[0], &tmp[1]));
}
Option& OptionContainer::add_option(const fextl::string& opt1, const fextl::string& opt2) {
  const fextl::string tmp[2] = { opt1, opt2 };
  return add_option(fextl::vector<fextl::string>(&tmp[0], &tmp[2]));
}
Option& OptionContainer::add_option(const fextl::string& opt1, const fextl::string& opt2, const fextl::string& opt3) {
  const fextl::string tmp[3] = { opt1, opt2, opt3 };
  return add_option(fextl::vector<fextl::string>(&tmp[0], &tmp[3]));
}
Option& OptionContainer::add_option(const fextl::vector<fextl::string>& v) {
  _opts.resize(_opts.size()+1, Option(get_parser()));
  Option& option = _opts.back();
  fextl::string dest_fallback;
  for (fextl::vector<fextl::string>::const_iterator it = v.begin(); it != v.end(); ++it) {
    if (it->substr(0,2) == "--") {
      const fextl::string s = it->substr(2);
      if (option.dest() == "")
        option.dest(str_replace(s, "-", "_"));
      option._long_opts.insert(s);
      _optmap_l[s] = &option;
    } else {
      const fextl::string s = it->substr(1,1);
      if (dest_fallback == "")
        dest_fallback = s;
      option._short_opts.insert(s);
      _optmap_s[s] = &option;
    }
  }
  if (option.dest() == "")
    option.dest(dest_fallback);
  return option;
}
fextl::string OptionContainer::format_option_help(unsigned int indent /* = 2 */) const {
  fextl::stringstream ss;

  if (_opts.empty())
    return ss.str();

  for (fextl::list<Option>::const_iterator it = _opts.begin(); it != _opts.end(); ++it) {
    if (it->help() != SUPPRESS_HELP)
      ss << it->format_help(indent);
  }

  return ss.str();
}
////////// } class OptionContainer //////////

////////// class OptionParser { //////////
OptionParser::OptionParser() :
  OptionContainer(),
  _usage(_("%prog [options]")),
  _add_help_option(true),
  _add_version_option(true),
  _interspersed_args(true) {}

OptionParser& OptionParser::add_option_group(const OptionGroup& group) {
  for (fextl::list<Option>::const_iterator oit = group._opts.begin(); oit != group._opts.end(); ++oit) {
    const Option& option = *oit;
    for (fextl::set<fextl::string>::const_iterator it = option._short_opts.begin(); it != option._short_opts.end(); ++it)
      _optmap_s[*it] = &option;
    for (fextl::set<fextl::string>::const_iterator it = option._long_opts.begin(); it != option._long_opts.end(); ++it)
      _optmap_l[*it] = &option;
  }
  _groups.push_back(&group);
  return *this;
}

const Option& OptionParser::lookup_short_opt(const fextl::string& opt) const {
  optMap::const_iterator it = _optmap_s.find(opt);
  if (it == _optmap_s.end())
    error(_("no such option") + fextl::string(": -") + opt);
  return *it->second;
}

void OptionParser::handle_short_opt(const fextl::string& opt, const fextl::string& arg) {

  _parsed.emplace_back(fextl::string("-") + opt);
  _remaining.pop_front();
  fextl::string value;

  const Option& option = lookup_short_opt(opt);
  if (option._nargs == 1) {
    value = arg.substr(2);
    if (value == "") {
      if (_remaining.empty()) {
        if (option._optional_value) {
          value = option.get_default();
          _parsed.emplace_back(value);
        }
        else {
          error("-" + opt + " " + _("option requires an argument"));
        }
      }
      else {
        value = _remaining.front();
        _remaining.pop_front();
        _parsed.emplace_back(value);
      }
    }
  } else {
    if (arg.length() > 2)
      _remaining.push_front(fextl::string("-") + arg.substr(2));
  }

  process_opt(option, fextl::string("-") + opt, value);
}

const Option& OptionParser::lookup_long_opt(const fextl::string& opt) const {

  fextl::list<fextl::string> matching;
  for (optMap::const_iterator it = _optmap_l.begin(); it != _optmap_l.end(); ++it) {
    if (it->first.compare(0, opt.length(), opt) == 0) {
      matching.push_back(it->first);
      if (it->first.length() == opt.length())
        break;
    }
  }
  if (matching.size() > 1) {
    fextl::string x = str_join_trans(", ", matching.begin(), matching.end(), str_wrap("--", ""));
    error(_("ambiguous option") + fextl::string(": --") + opt + " (" + x + "?)");
  }
  if (matching.size() == 0)
    error(_("no such option") + fextl::string(": --") + opt);

  return *_optmap_l.find(matching.front())->second;
}

void OptionParser::handle_long_opt(const fextl::string& optstr) {

  _remaining.pop_front();
  fextl::string opt, value;

  size_t delim = optstr.find("=");
  if (delim != fextl::string::npos) {
    opt = optstr.substr(0, delim);
    value = optstr.substr(delim+1);
  } else
    opt = optstr;

  const Option& option = lookup_long_opt(opt);
  if (option._nargs == 1 and delim == fextl::string::npos) {
    if (not _remaining.empty()) {
      value = _remaining.front();
      _remaining.pop_front();
    }
  }

  if (option._nargs == 1 and value == "")
    error("--" + opt + " " + _("option requires an argument"));

  process_opt(option, fextl::string("--") + opt, value);
}

Values& OptionParser::parse_args(const int argc, char const* const* const argv) {
  if (prog() == "")
    prog(basename(argv[0]));

  _parsed.emplace_back(argv[0]);
  return parse_args(&argv[1], &argv[argc]);
}
Values& OptionParser::parse_args(const fextl::vector<fextl::string>& v) {

  _remaining.assign(v.begin(), v.end());

  if (add_help_option()) {
    add_option("-h", "--help") .action("help") .help(_("show this help message and exit"));
    _opts.splice(_opts.begin(), _opts, --(_opts.end()));
  }
  if (add_version_option() and version() != "") {
    add_option("--version") .action("version") .help(_("show program's version number and exit"));
    _opts.splice(_opts.begin(), _opts, --(_opts.end()));
  }

  while (not _remaining.empty()) {
    const fextl::string arg = _remaining.front();

    if (arg == "--") {
      _remaining.pop_front();
      break;
    }

    if (arg.substr(0,2) == "--") {
      _parsed.emplace_back(arg);
      handle_long_opt(arg.substr(2));
    } else if (arg.substr(0,1) == "-" and arg.length() > 1) {
      handle_short_opt(arg.substr(1,1), arg);
    } else {
      _remaining.pop_front();
      _leftover.push_back(arg);
      if (not interspersed_args())
        break;
    }
  }
  while (not _remaining.empty()) {
    const fextl::string arg = _remaining.front();
    _remaining.pop_front();
    _leftover.push_back(arg);
  }

  for (fextl::list<Option>::const_iterator it = _opts.begin(); it != _opts.end(); ++it) {
    if (it->get_default() != "" and not _values.is_set(it->dest()))
      _values[it->dest()] = it->get_default();
  }

  for (fextl::list<OptionGroup const*>::iterator group_it = _groups.begin(); group_it != _groups.end(); ++group_it) {
    for (fextl::list<Option>::const_iterator it = (*group_it)->_opts.begin(); it != (*group_it)->_opts.end(); ++it) {
      if (it->get_default() != "" and not _values.is_set(it->dest()))
        _values[it->dest()] = it->get_default();
    }
  }

  return _values;
}

void OptionParser::process_opt(const Option& o, const fextl::string& opt, const fextl::string& value) {
  if (o.action() == "store") {
    fextl::string err = o.check_type(opt, value);
    if (err != "")
      error(err);
    _values[o.dest()] = value;
    _values.is_set_by_user(o.dest(), true);
  }
  else if (o.action() == "store_const") {
    _values[o.dest()] = o.get_const();
    _values.is_set_by_user(o.dest(), true);
  }
  else if (o.action() == "store_true") {
    _values[o.dest()] = "1";
    _values.is_set_by_user(o.dest(), true);
  }
  else if (o.action() == "store_false") {
    _values[o.dest()] = "0";
    _values.is_set_by_user(o.dest(), true);
  }
  else if (o.action() == "append") {
    fextl::string err = o.check_type(opt, value);
    if (err != "")
      error(err);
    _values[o.dest()] = value;
    _values.all(o.dest()).push_back(value);
    _values.is_set_by_user(o.dest(), true);
  }
  else if (o.action() == "append_const") {
    _values[o.dest()] = o.get_const();
    _values.all(o.dest()).push_back(o.get_const());
    _values.is_set_by_user(o.dest(), true);
  }
  else if (o.action() == "count") {
    _values[o.dest()] = str_inc(_values[o.dest()]);
    _values.is_set_by_user(o.dest(), true);
  }
  else if (o.action() == "help") {
    print_help();
    std::exit(0);
  }
  else if (o.action() == "version") {
    print_version();
    std::exit(0);
  }
  else if (o.action() == "callback" && o.callback()) {
    fextl::string err = o.check_type(opt, value);
    if (err != "")
      error(err);
    (*o.callback())(o, opt, value, *this);
  }
}

fextl::string OptionParser::format_help() const {
  fextl::stringstream ss;

  if (usage() != SUPPRESS_USAGE)
    ss << get_usage() << std::endl;

  if (description() != "")
    ss << str_format(description(), 0, cols()) << std::endl;

  ss << _("Options") << ":" << std::endl;
  ss << format_option_help();

  for (fextl::list<OptionGroup const*>::const_iterator it = _groups.begin(); it != _groups.end(); ++it) {
    const OptionGroup& group = **it;
    ss << std::endl << "  " << group.title() << ":" << std::endl;
    if (group.description() != "") {
      unsigned int malus = 4; // Python seems to not use full length
      ss << str_format(group.description(), 4, cols() - malus) << std::endl;
    }
    ss << group.format_option_help(4);
  }

  if (epilog() != "")
    ss << std::endl << str_format(epilog(), 0, cols());

  return ss.str();
}
void OptionParser::print_help() const {
  std::cout << format_help();
}

void OptionParser::set_usage(const fextl::string& u) {
  fextl::string lower = u;
  transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower.compare(0, 7, "usage: ") == 0)
    _usage = u.substr(7);
  else
    _usage = u;
}
fextl::string OptionParser::format_usage(const fextl::string& u) const {
  fextl::stringstream ss;
  ss << _("Usage") << ": " << u << std::endl;
  return ss.str();
}
fextl::string OptionParser::get_usage() const {
  if (usage() == SUPPRESS_USAGE)
    return fextl::string("");
  return format_usage(str_replace(usage(), "%prog", prog()));
}
void OptionParser::print_usage(std::ostream& out) const {
  fextl::string u = get_usage();
  if (u != "")
    out << u << std::endl;
}
void OptionParser::print_usage() const {
  print_usage(std::cout);
}

fextl::string OptionParser::get_version() const {
  return str_replace(_version, "%prog", prog());
}
void OptionParser::print_version(std::ostream& out) const {
  out << get_version() << std::endl;
}
void OptionParser::print_version() const {
  print_version(std::cout);
}

void OptionParser::exit() const {
  std::exit(EXIT_FAILURE);
}
void OptionParser::error(const fextl::string& msg) const {
  print_usage(std::cerr);
  std::cerr << prog() << ": " << _("error") << ": " << msg << std::endl;
  exit();
}
////////// } class OptionParser //////////

////////// class Values { //////////
std::optional<const fextl::string*> Values::operator[] (const fextl::string& d) const {
  strMap::const_iterator it = _map.find(d);
  return (it != _map.end()) ? std::optional(&it->second) : std::nullopt;
}
void Values::is_set_by_user(const fextl::string& d, bool yes) {
  if (yes)
    _userSet.insert(d);
  else
    _userSet.erase(d);
}
////////// } class Values //////////

////////// class Option { //////////
fextl::string Option::check_type(const fextl::string& opt, const fextl::string& val) const {
  fextl::istringstream ss(val);
  fextl::stringstream err;

  if (type() == "int" || type() == "long") {
    long t;
    if (not (ss >> t))
      err << _("option") << " " << opt << ": " << _("invalid integer value") << ": '" << val << "'";
  }
  else if (type() == "float" || type() == "double") {
    double t;
    if (not (ss >> t))
      err << _("option") << " " << opt << ": " << _("invalid floating-point value") << ": '" << val << "'";
  }
  else if (type() == "choice") {
    if (find(choices().begin(), choices().end(), val) == choices().end()) {
      fextl::list<fextl::string> tmp = choices();
      transform(tmp.begin(), tmp.end(), tmp.begin(), str_wrap("'"));
      err << _("option") << " " << opt << ": " << _("invalid choice") << ": '" << val << "'"
        << " (" << _("choose from") << " " << str_join(", ", tmp.begin(), tmp.end()) << ")";
    }
  }
  else if (type() == "complex") {
    std::complex<double> t;
    if (not (ss >> t))
      err << _("option") << " " << opt << ": " << _("invalid complex value") << ": '" << val << "'";
  }

  return err.str();
}

fextl::string Option::format_option_help(unsigned int indent /* = 2 */) const {

  fextl::string mvar_short, mvar_long;
  if (nargs() == 1) {
    fextl::string mvar = metavar();
    if (mvar == "") {
      mvar = dest();
      transform(mvar.begin(), mvar.end(), mvar.begin(), ::toupper);
     }
    if (_optional_value) {
      mvar_short = " [" + mvar + "]";
      mvar_long = "[=" + mvar + "]";
    }
    else {
      mvar_short = " " + mvar;
      mvar_long = "=" + mvar;
    }
  }

  fextl::stringstream ss;
  ss << fextl::string(indent, ' ');

  if (not _short_opts.empty()) {
    ss << str_join_trans(", ", _short_opts.begin(), _short_opts.end(), str_wrap("-", mvar_short));
    if (not _long_opts.empty())
      ss << ", ";
  }
  if (not _long_opts.empty())
    ss << str_join_trans(", ", _long_opts.begin(), _long_opts.end(), str_wrap("--", mvar_long));

  return ss.str();
}

fextl::string Option::format_help(unsigned int indent /* = 2 */) const {
  fextl::stringstream ss;
  fextl::string h = format_option_help(indent);
  unsigned int width = cols();
  unsigned int opt_width = std::min(width*3/10, 36u);
  bool indent_first = false;
  ss << h;
  // if the option list is too long, start a new paragraph
  if (h.length() >= (opt_width-1)) {
    ss << std::endl;
    indent_first = true;
  } else {
    ss << fextl::string(opt_width - h.length(), ' ');
    if (help() == "")
      ss << std::endl;
  }
  if (help() != "") {
    fextl::string help_str = (get_default() != "") ? str_replace(help(), "%default", get_default()) : help();
    ss << str_format(help_str, opt_width, width, false, indent_first);
  }
  return ss.str();
}

Option& Option::action(const fextl::string& a) {
  _action = a;
  if (a == "store_const" || a == "store_true" || a == "store_false" ||
      a == "append_const" || a == "count" || a == "help" || a == "version") {
    nargs(0);
  } else if (a == "callback") {
    nargs(0);
    type("");
  }
  return *this;
}


Option& Option::type(const fextl::string& t) {
  _type = t;
  nargs((t == "") ? 0 : 1);
  return *this;
}

const fextl::string& Option::get_default() const {
  strMap::const_iterator it = _parser._defaults.find(dest());
  if (it != _parser._defaults.end())
    return it->second;
  else
    return _default;
}
////////// } class Option //////////

}
