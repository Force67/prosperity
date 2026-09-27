// Copyright 2010 Dolphin Emulator Project
// Licensed under GPLv2+
// Refer to the license.txt file included.

#pragma once

// a simple lockless thread-safe,
// single reader, single writer queue

#include "base/arch.h"
#include "base/atomic.h"
#include "base/memory/move.h"
#include "base/threading/condition_variable.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"

namespace common {
template <typename T>
class SPSCQueue {
 public:
  SPSCQueue() { write_ptr_ = read_ptr_ = new ElementPtr(); }
  ~SPSCQueue() {
    // this will empty out the whole queue
    delete read_ptr_;
  }

  mem_size Size() const { return size_.load(); }

  bool Empty() const { return Size() == 0; }

  T& Front() const { return read_ptr_->current; }

  template <typename Arg>
  void Push(Arg&& t) {
    // create the element, add it to the queue
    write_ptr_->current = base::forward<Arg>(t);
    // set the next pointer to a new element ptr
    // then advance the write pointer
    ElementPtr* new_ptr = new ElementPtr();
    write_ptr_->next.store(new_ptr, base::memory_order_release);
    write_ptr_ = new_ptr;
    // publish size before notifying, under cv_mutex, or PopWait can check its
    // predicate, miss this element, and sleep through the wakeup (which hung
    // process exit when the lost entry was the logger's final one)
    ++size_;
    {
      base::LockGuard<base::Mutex> lock{cv_mutex_};
    }
    cv_.NotifyOne();
  }

  void Pop() {
    --size_;

    ElementPtr* tmpptr = read_ptr_;
    // advance the read pointer
    read_ptr_ = tmpptr->next.load();
    // set the next element to nullptr to stop the recursive deletion
    tmpptr->next.store(nullptr);
    delete tmpptr;  // this also deletes the element
  }

  bool Pop(T& t) {
    if (Empty())
      return false;

    --size_;

    ElementPtr* tmpptr = read_ptr_;
    read_ptr_ = tmpptr->next.load(base::memory_order_acquire);
    t = base::move(tmpptr->current);
    tmpptr->next.store(nullptr);
    delete tmpptr;
    return true;
  }

  T PopWait() {
    if (Empty()) {
      base::UniqueLock<base::Mutex> lock{cv_mutex_};
      cv_.Wait(lock, [this]() { return !Empty(); });
    }
    T t;
    Pop(t);
    return t;
  }

  // not thread-safe
  void Clear() {
    size_.store(0);
    delete read_ptr_;
    write_ptr_ = read_ptr_ = new ElementPtr();
  }

 private:
  // stores a pointer to element
  // and a pointer to the next ElementPtr
  class ElementPtr {
   public:
    ElementPtr() {}
    ~ElementPtr() {
      ElementPtr* next_ptr = next.load();

      if (next_ptr)
        delete next_ptr;
    }

    T current;
    base::Atomic<ElementPtr*> next{nullptr};
  };

  ElementPtr* write_ptr_;
  ElementPtr* read_ptr_;
  base::Atomic<mem_size> size_{0};
  base::Mutex cv_mutex_;
  base::ConditionVariable cv_;
};

// a simple thread-safe,
// single reader, multiple writer queue

template <typename T>
class MPSCQueue {
 public:
  mem_size Size() const { return spsc_queue_.Size(); }

  bool Empty() const { return spsc_queue_.Empty(); }

  T& Front() const { return spsc_queue_.Front(); }

  template <typename Arg>
  void Push(Arg&& t) {
    base::LockGuard<base::Mutex> lock{write_lock_};
    spsc_queue_.Push(t);
  }

  void Pop() { return spsc_queue_.Pop(); }

  bool Pop(T& t) { return spsc_queue_.Pop(t); }

  T PopWait() { return spsc_queue_.PopWait(); }

  // not thread-safe
  void Clear() { spsc_queue_.Clear(); }

 private:
  SPSCQueue<T> spsc_queue_;
  base::Mutex write_lock_;
};
}  // namespace common