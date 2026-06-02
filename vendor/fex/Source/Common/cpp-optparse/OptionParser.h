/**
 * Copyright (C) 2010 Johannes Weißl <jargon@molb.org>
 * License: MIT License
 * URL: https://github.com/weisslj/cpp-optparse
 */

#ifndef OPTIONPARSER_H_
#define OPTIONPARSER_H_

#include <FEXCore/fextl/list.h>
#include <FEXCore/fextl/map.h>
#include <FEXCore/fextl/set.h>
#include <FEXCore/fextl/string.h>
#include <FEXCore/fextl/sstream.h>
#include <FEXCore/fextl/vector.h>

#include <map>
#include <iostream>
#include <optional>

namespace optparse {

class OptionParser;
class OptionGroup;
class Option;
class Values;
class Value;
class Callback;

typedef fextl::map<fextl::string,fextl::string> strMap;
typedef fextl::map<fextl::string,fextl::list<fextl::string> > lstMap;
typedef fextl::map<fextl::string,Option const*> optMap;

const char* const SUPPRESS_HELP = "SUPPRESS" "HELP";
const char* const SUPPRESS_USAGE = "SUPPRESS" "USAGE";

//! Class for automatic conversion from string -> anytype
class Value {
  public:
    Value() : str(), valid(false) {}
    Value(const fextl::string& v) : str(v), valid(true) {}
    operator const char*() { return str.c_str(); }
    operator bool() { bool t; return (valid && (fextl::istringstream(str) >> t)) ? t : false; }
    operator short() { short t; return (valid && (fextl::istringstream(str) >> t)) ? t : 0; }
    operator unsigned short() { unsigned short t; return (valid && (fextl::istringstream(str) >> t)) ? t : 0; }
    operator int() { int t; return (valid && (fextl::istringstream(str) >> t)) ? t : 0; }
    operator unsigned int() { unsigned int t; return (valid && (fextl::istringstream(str) >> t)) ? t : 0; }
    operator long() { long t; return (valid && (fextl::istringstream(str) >> t)) ? t : 0; }
    operator unsigned long() { unsigned long t; return (valid && (fextl::istringstream(str) >> t)) ? t : 0; }
    operator float() { float t; return (valid && (fextl::istringstream(str) >> t)) ? t : 0; }
    operator double() { double t; return (valid && (fextl::istringstream(str) >> t)) ? t : 0; }
    operator long double() { long double t; return (valid && (fextl::istringstream(str) >> t)) ? t : 0; }
 private:
    const fextl::string str;
    bool valid;
};

class Values {
  public:
    Values() : _map() {}
    std::optional<const fextl::string*> operator[] (const fextl::string& d) const;
    fextl::string& operator[] (const fextl::string& d) { return _map[d]; }
    bool is_set(const fextl::string& d) const { return _map.find(d) != _map.end(); }
    bool is_set_by_user(const fextl::string& d) const { return _userSet.find(d) != _userSet.end(); }
    void is_set_by_user(const fextl::string& d, bool yes);
    Value get(const fextl::string& d) const { return (is_set(d)) ? Value(*(*this)[d].value()) : Value(); }

    typedef fextl::list<fextl::string>::iterator iterator;
    typedef fextl::list<fextl::string>::const_iterator const_iterator;
    fextl::list<fextl::string>& all(const fextl::string& d) { return _appendMap[d]; }
    const fextl::list<fextl::string>& all(const fextl::string& d) const { return _appendMap.find(d)->second; }

  private:
    strMap _map;
    lstMap _appendMap;
    fextl::set<fextl::string> _userSet;
};

class Option {
  public:
    Option(const OptionParser& p) :
      _parser(p), _action("store"), _type("string"), _nargs(1), _callback(0) {}
    virtual ~Option() {}

    Option& action(const fextl::string& a);
    Option& type(const fextl::string& t);
    Option& dest(const fextl::string& d) { _dest = d; return *this; }
    Option& set_default(const fextl::string& d) { _default = d; return *this; }
    template<typename T>
    Option& set_default(T t) { fextl::ostringstream ss; ss << t; _default = ss.str(); return *this; }
    Option& nargs(size_t n) { _nargs = n; return *this; }
    Option& set_optional_value (bool v) { _optional_value = v; return *this; }
    Option& set_const(const fextl::string& c) { _const = c; return *this; }
    template<typename InputIterator>
    Option& choices(InputIterator begin, InputIterator end) {
      _choices.assign(begin, end); type("choice"); return *this;
    }
#if __cplusplus >= 201103L
    Option& choices(std::initializer_list<fextl::string> ilist) {
      _choices.assign(ilist); type("choice"); return *this;
    }
#endif
    Option& help(const fextl::string& h) { _help = h; return *this; }
    Option& metavar(const fextl::string& m) { _metavar = m; return *this; }
    Option& callback(Callback& c) { _callback = &c; return *this; }

    const fextl::string& action() const { return _action; }
    const fextl::string& type() const { return _type; }
    const fextl::string& dest() const { return _dest; }
    const fextl::string& get_default() const;
    size_t nargs() const { return _nargs; }
    const fextl::string& get_const() const { return _const; }
    const fextl::list<fextl::string>& choices() const { return _choices; }
    const fextl::string& help() const { return _help; }
    const fextl::string& metavar() const { return _metavar; }
    Callback* callback() const { return _callback; }

  private:
    fextl::string check_type(const fextl::string& opt, const fextl::string& val) const;
    fextl::string format_option_help(unsigned int indent = 2) const;
    fextl::string format_help(unsigned int indent = 2) const;

    const OptionParser& _parser;

    fextl::set<fextl::string> _short_opts;
    fextl::set<fextl::string> _long_opts;

    bool _optional_value;
    fextl::string _action;
    fextl::string _type;
    fextl::string _dest;
    fextl::string _default;
    size_t _nargs;
    fextl::string _const;
    fextl::list<fextl::string> _choices;
    fextl::string _help;
    fextl::string _metavar;
    Callback* _callback;

    friend class OptionContainer;
    friend class OptionParser;
};

class OptionContainer {
  public:
    OptionContainer(const fextl::string& d = "") : _description(d) {}
    virtual ~OptionContainer() {}

    virtual OptionContainer& description(const fextl::string& d) { _description = d; return *this; }
    virtual const fextl::string& description() const { return _description; }

    Option& add_option(const fextl::string& opt);
    Option& add_option(const fextl::string& opt1, const fextl::string& opt2);
    Option& add_option(const fextl::string& opt1, const fextl::string& opt2, const fextl::string& opt3);
    Option& add_option(const fextl::vector<fextl::string>& opt);

    fextl::string format_option_help(unsigned int indent = 2) const;

  protected:
    fextl::string _description;

    fextl::list<Option> _opts;
    optMap _optmap_s;
    optMap _optmap_l;

  private:
    virtual const OptionParser& get_parser() = 0;
};

class OptionParser : public OptionContainer {
  public:
    OptionParser();
    virtual ~OptionParser() {}

    OptionParser& usage(const fextl::string& u) { set_usage(u); return *this; }
    OptionParser& version(const fextl::string& v) { _version = v; return *this; }
    OptionParser& description(const fextl::string& d) { _description = d; return *this; }
    OptionParser& add_help_option(bool h) { _add_help_option = h; return *this; }
    OptionParser& add_version_option(bool v) { _add_version_option = v; return *this; }
    OptionParser& prog(const fextl::string& p) { _prog = p; return *this; }
    OptionParser& epilog(const fextl::string& e) { _epilog = e; return *this; }
    OptionParser& set_defaults(const fextl::string& dest, const fextl::string& val) {
      _defaults[dest] = val; return *this;
    }
    template<typename T>
    OptionParser& set_defaults(const fextl::string& dest, T t) { fextl::ostringstream ss; ss << t; _defaults[dest] = ss.str(); return *this; }
    OptionParser& enable_interspersed_args() { _interspersed_args = true; return *this; }
    OptionParser& disable_interspersed_args() { _interspersed_args = false; return *this; }
    OptionParser& add_option_group(const OptionGroup& group);

    const fextl::string& usage() const { return _usage; }
    const fextl::string& version() const { return _version; }
    const fextl::string& description() const { return _description; }
    bool add_help_option() const { return _add_help_option; }
    bool add_version_option() const { return _add_version_option; }
    const fextl::string& prog() const { return _prog; }
    const fextl::string& epilog() const { return _epilog; }
    bool interspersed_args() const { return _interspersed_args; }

    Values& parse_args(int argc, char const* const* argv);
    Values& parse_args(const fextl::vector<fextl::string>& args);
    template<typename InputIterator>
    Values& parse_args(InputIterator begin, InputIterator end) {
      return parse_args(fextl::vector<fextl::string>(begin, end));
    }

    const fextl::list<fextl::string>& args() const { return _leftover; }
    fextl::vector<fextl::string> args() {
      return fextl::vector<fextl::string>(_leftover.begin(), _leftover.end());
    }

    fextl::vector<fextl::string> parsed_args() {
      return fextl::vector<fextl::string>(_parsed.begin(), _parsed.end());
    }

    fextl::string format_help() const;
    void print_help() const;

    void set_usage(const fextl::string& u);
    fextl::string get_usage() const;
    void print_usage(std::ostream& out) const;
    void print_usage() const;

    fextl::string get_version() const;
    void print_version(std::ostream& out) const;
    void print_version() const;

    void error(const fextl::string& msg) const;
    void exit() const;

  private:
    const OptionParser& get_parser() { return *this; }
    const Option& lookup_short_opt(const fextl::string& opt) const;
    const Option& lookup_long_opt(const fextl::string& opt) const;

    void handle_short_opt(const fextl::string& opt, const fextl::string& arg);
    void handle_long_opt(const fextl::string& optstr);

    void process_opt(const Option& option, const fextl::string& opt, const fextl::string& value);

    fextl::string format_usage(const fextl::string& u) const;

    fextl::string _usage;
    fextl::string _version;
    bool _add_help_option;
    bool _add_version_option;
    fextl::string _prog;
    fextl::string _epilog;
    bool _interspersed_args;

    Values _values;

    strMap _defaults;
    fextl::list<OptionGroup const*> _groups;

    fextl::list<fextl::string> _remaining;
    fextl::list<fextl::string> _leftover;
    fextl::list<fextl::string> _parsed;

    friend class Option;
};

class OptionGroup : public OptionContainer {
  public:
    OptionGroup(const OptionParser& p, const fextl::string& t, const fextl::string& d = "") :
      OptionContainer(d), _parser(p), _title(t) {}
    virtual ~OptionGroup() {}

    OptionGroup& title(const fextl::string& t) { _title = t; return *this; }
    const fextl::string& title() const { return _title; }

  private:
    const OptionParser& get_parser() { return _parser; }

    const OptionParser& _parser;
    fextl::string _title;

  friend class OptionParser;
};

class Callback {
public:
  virtual void operator() (const Option& option, const fextl::string& opt, const fextl::string& val, const OptionParser& parser) = 0;
  virtual ~Callback() {}
};

}

#endif
