/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PTI_MEMBLOCK_INCLUDED
#define PTI_MEMBLOCK_INCLUDED

#include <cstddef>
#include <ParTI/memnode.hpp>
#include <ParTI/session.hpp>

namespace pti {

template <typename T>
struct MemBlock {

private:

    int last_node;

    T** pointers;

public:

    explicit MemBlock() {
        int num_nodes = session.mem_nodes.size();
        last_node = -1;
        pointers = new T* [num_nodes]();
    }

    MemBlock(MemBlock&& other) {
        last_node = other.last_node;
        pointers = other.pointers;
        other.pointers = nullptr;
    }

    MemBlock& operator= (MemBlock&& other) {
        if(pointers) {
            int num_nodes = session.mem_nodes.size();
            for(int i = 0; i < num_nodes; ++i) {
                if(pointers[i]) {
                    session.mem_nodes[i]->free(pointers[i]);
                }
            }
        }
        delete[] pointers;

        last_node = other.last_node;
        pointers = other.pointers;
        other.pointers = nullptr;
        return *this;
    }

    ~MemBlock() {
        if(pointers) {
            int num_nodes = session.mem_nodes.size();
            for(int i = 0; i < num_nodes; ++i) {
                if(pointers[i]) {
                    session.mem_nodes[i]->free(pointers[i]);
                }
            }
        }
        delete[] pointers;
    }

    /// Allocate an element on `node` and set `last_node` to `node`.
    /// The element are not initialized.
    void allocate(int node) {
        // If you need to call the constructor, allocate it on CPU, then
        //     new (get(CPU_node)) T();
        // to invoke the constructor properly.
        if(!pointers[node]) {
            pointers[node] = reinterpret_cast<T*>(session.mem_nodes[node]->malloc(sizeof (T)));
        }
        if(last_node == -1) {
            last_node = node;
        }
    }

    /// If `node` is not `last_node`, free memory allocated on `node`.
    void free(int node) {
        if(node != last_node && pointers[node]) {
            session.mem_nodes[node]->free(pointers[node]);
            pointers[node] = nullptr;
        }
    }

    /// If an element is initialized, copy the element on `last_node` to `node`, then set `last_node` to `node`.
    void copy_to(int node) {
        if(node != last_node && last_node != -1) {
            allocate(node);
            session.mem_nodes[last_node]->memcpy_to(pointers[node], *session.mem_nodes[node], pointers[last_node], sizeof (T));
            last_node = node;
        }
    }

    /// Copy the pointer from `last_node` to `node`, then get pointer on `node`.
    T* operator() (int node) {
        copy_to(node);
        return pointers[node];
    }

    /// Get pointer on `node`.
    T* ptr(int node) const {
        return pointers[node];
    }

    /// Set `last_node` to `node`.
    void mark_dirty(int node) {
        last_node = node;
    }

    /// Create a clone of this block
    MemBlock clone(int node) {
        MemBlock result;
        result.allocate(node);
        session.mem_nodes[last_node]->memcpy_to(result.pointers[node], *session.mem_nodes[node], pointers[last_node], sizeof (T));
        return result;
    }

};

template <typename T>
struct MemBlock<T[]> {

private:

    int last_node;

    T** pointers;

    size_t* sizes;

public:

    explicit MemBlock() {
        int num_nodes = session.mem_nodes.size();
        last_node = -1;
        pointers = new T* [num_nodes]();
        sizes = new size_t [num_nodes]();
    }

    MemBlock(MemBlock&& other) {
        last_node = other.last_node;
        pointers = other.pointers;
        sizes = other.sizes;
        other.pointers = nullptr;
        other.sizes = nullptr;
    }

    MemBlock& operator= (MemBlock&& other) {
        if(pointers) {
            int num_nodes = session.mem_nodes.size();
            for(int i = 0; i < num_nodes; ++i) {
                if(pointers[i]) {
                    session.mem_nodes[i]->free(pointers[i]);
                }
            }
        }
        delete[] pointers;
        delete[] sizes;

        last_node = other.last_node;
        pointers = other.pointers;
        sizes = other.sizes;
        other.pointers = nullptr;
        other.sizes = nullptr;
        return *this;
    }

    ~MemBlock() {
        if(pointers) {
            int num_nodes = session.mem_nodes.size();
            for(int i = 0; i < num_nodes; ++i) {
                if(pointers[i]) {
                    session.mem_nodes[i]->free(pointers[i]);
                }
            }
        }
        delete[] pointers;
        delete[] sizes;
    }

    /// Allocate `size` elements on `node` and set `last_node` to `node`.
    /// Elements are not initialized.
    void allocate(int node, size_t size) {
        // If you need to call the constructor, allocate it on CPU, then
        //     new (get(CPU_node)) T [size]();
        // to invoke the constructor properly.
        if(sizes[node] != size) {
            if(pointers[node]) {
                session.mem_nodes[node]->free(pointers[node]);
            }
            pointers[node] = reinterpret_cast<T*>(session.mem_nodes[node]->malloc(size * sizeof (T)));
            sizes[node] = size;
        }
        if(last_node == -1) {
            last_node = node;
        }
    }

    /// Allocate `size` elements on `node` and copy as many elements as possible from `last_node`, then set `last_node` to `node`.
    /// Newly allocated elements are not initialized.
    void resize(int node, size_t size) {
        if(!pointers[node]) {
            pointers[node] = reinterpret_cast<T*>(session.mem_nodes[node]->malloc(size * sizeof (T)));
            sizes[node] = size;
        } else {
            T* new_ptr = reinterpret_cast<T*>(session.mem_nodes[node]->malloc(size * sizeof (T)));
            session.mem_nodes[last_node]->memcpy_to(new_ptr, *session.mem_nodes[node], pointers[last_node], (size < sizes[node] ? size : sizes[node]) * sizeof (T));
            session.mem_nodes[node]->free(pointers[node]);
            pointers[node] = new_ptr;
            sizes[node] = size;
        }
        if(last_node == -1) {
            last_node = node;
        }
    }

    /// If `node` is not `last_node`, free memory allocated on `node`.
    void free(int node) {
        if(node != last_node && pointers[node]) {
            session.mem_nodes[node]->free(pointers[node]);
            pointers[node] = nullptr;
            sizes[node] = 0;
        }
    }

    /// Copy all elements on `last_node` to `node`, then set `last_node` to `node`.
    void copy_to(int node) {
        if(node != last_node) {
            if(last_node != -1 && sizes[last_node] != 0) {
                allocate(node, sizes[last_node]);
                session.mem_nodes[last_node]->memcpy_to(pointers[node], *session.mem_nodes[node], pointers[last_node], sizes[last_node] * sizeof (T));
                sizes[node] = sizes[last_node];
            } else {
                this->free(node);
                sizes[node] = 0;
            }
            last_node = node;
        }
    }

    /// Copy the pointer from `last_node` to `node`, then get pointer on `node`.
    T* operator() (int node) {
        copy_to(node);
        return pointers[node];
    }

    /// Get pointer on `node`.
    T* ptr(int node) const {
        return pointers[node];
    }

    /// Get size on `last_node`.
    size_t size() const {
        return last_node != -1 ? sizes[last_node] : 0;
    }

    /// Set `last_node` to `node`.
    void mark_dirty(int node) {
        last_node = node;
    }

    /// Create a clone of this block
    MemBlock clone(int node) {
        MemBlock result;
        result.allocate(node, sizes[last_node]);
        session.mem_nodes[last_node]->memcpy_to(result.pointers[node], *session.mem_nodes[node], pointers[last_node], sizes[last_node] * sizeof (T));
        return result;
    }

};

}

#endif
