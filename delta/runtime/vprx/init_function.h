#pragma once

/*
    Copyright 2018 Brick

    Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software
    and associated documentation files (the "Software"), to deal in the Software
   without restriction,
    including without limitation the rights to use, copy, modify, merge,
   publish, distribute,
    sublicense, and/or sell copies of the Software, and to permit persons to
   whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
   all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING
    BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
   PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
   LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
   OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

namespace runtime {
class InitFunction {
 public:
  using CallbackT = void (*)();

 private:
  InitFunction* next_{nullptr};
  CallbackT callback_{nullptr};

  InitFunction(InitFunction*& parent, CallbackT callback) noexcept;

 public:
  InitFunction(CallbackT callback) noexcept;
  InitFunction(InitFunction& parent, CallbackT callback) noexcept;

  InitFunction(const InitFunction&) = delete;
  InitFunction(InitFunction&&) = delete;

  static InitFunction*& ROOT() noexcept;

  static mem_size Init();
};

inline InitFunction::InitFunction(InitFunction*& parent,
                                  CallbackT callback) noexcept
    : next_(parent), callback_(callback) {
  parent = this;
}

inline InitFunction::InitFunction(CallbackT callback) noexcept
    : InitFunction(ROOT(), callback) {}

inline InitFunction::InitFunction(InitFunction& parent,
                                  CallbackT callback) noexcept
    : InitFunction(parent.next_, callback) {}

inline InitFunction*& InitFunction::ROOT() noexcept {
  static InitFunction* root{nullptr};

  return root;
}

inline mem_size InitFunction::Init() {
  mem_size total = 0;

  for (InitFunction* i = ROOT(); i;) {
    if (i->callback_) {
      i->callback_();
      i->callback_ = nullptr;

      ++total;
    }

    InitFunction* j = i->next_;
    i->next_ = nullptr;
    i = j;
  }

  ROOT() = nullptr;

  return total;
}
}  // namespace runtime