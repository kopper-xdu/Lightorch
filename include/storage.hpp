#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

class Storage
{
public:
    Storage(uint32_t size) : start_ptr_(malloc(size)) 
    { memset(start_ptr_, 0, size); }

    ~Storage() 
    { free(start_ptr_); }

    void *start_ptr_;
};