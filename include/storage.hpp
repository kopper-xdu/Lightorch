#pragma once

#include <cstdint>
#include <cstdlib>

class Storage
{
public:
    Storage(uint32_t size) : start_ptr_(malloc(size)) { }

    ~Storage() { free(start_ptr_); }

    void *start_ptr_;
};