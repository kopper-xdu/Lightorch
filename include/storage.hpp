#include <cstdint>
#include <cstdlib>

class Storage
{
public:
    Storage(uint32_t size) : data(malloc(size)) { }
    // Storage(const std::vector<uint32_t> &shape, uint32_t size) 
    // { 

    // }

    ~Storage() { free(data); }

    void *data;
};