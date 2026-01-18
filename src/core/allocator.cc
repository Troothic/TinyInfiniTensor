#include "core/allocator.h"
#include <utility>
// Trigger CI rebuild

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        
        // 优先查找够大的空闲块
        for(auto it = free_blocks.begin(); it != free_blocks.end(); it++)
        {
            if(it->second >= size)
            {
                size_t addr = it->first;
                if(it->second > size)
                {
                    free_blocks[addr + size] = it->second - size;
                }
                free_blocks.erase(it);
                return addr;
            }
        }
        
        // 检查末尾是否有空闲块可以扩展
        if (!free_blocks.empty())
        {
            auto it = std::prev(free_blocks.end());  // 最后一个空闲块
            if (it->first + it->second == used)  // 如果这个块在末尾
            {
                size_t addr = it->first;
                size_t extra = size - it->second;  // 需要额外分配的大小
                free_blocks.erase(it);
                used += extra;  // 扩展末尾
                if (used > peak) peak = used;
                return addr;
            }
        }
        
        // 从末尾分配新空间
        size_t addr = used;
        used += size;
        if(used > peak)
        {
            peak = used;
        }
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        free_blocks[addr] = size;
        auto it = free_blocks.find(addr); //it为free_blocks中addr的迭代器
        //前瞻
        if(it != free_blocks.begin())
        {
            auto prev = std::prev(it);
            size_t prev_addr = prev->first;
            size_t prev_size = prev->second;
            if(prev_addr + prev_size == addr)
            {
                addr = prev_addr;
                size += prev_size;
                free_blocks.erase(prev);     
                free_blocks.erase(it);           
                free_blocks[addr] = size;
            }
        }
        //后顾
        auto it2  = free_blocks.find(addr);
        auto next = std::next(it2);
        if(next != free_blocks.end())
        {
            size_t next_addr = next->first;
            size_t next_size = next->second;
            if(addr + size == next_addr)
            {
                size += next_size;
                free_blocks.erase(next);            
                free_blocks[addr] = size;
            }
        }


    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %zu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }
    //把任意大小“向上取整”到alignment的倍数
    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
