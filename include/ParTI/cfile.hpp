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

#ifndef PTI_CFILE_INCLUDED
#define PTI_CFILE_INCLUDED

#include <cassert>
#include <cstdio>
#include <ParTI/error.hpp>

namespace pti {

class CFile {

private:

    std::FILE* fp;

public:

    explicit CFile() {
        fp = nullptr;
    }

    explicit CFile(std::FILE* fp) {
        this->fp = fp;
    }

    explicit CFile(CFile const&) = delete;

    explicit CFile(CFile&& other) {
        fp = other.fp;
        other.fp = nullptr;
    }

    explicit CFile(char const* pathname, char const* mode) {
        fp = std::fopen(pathname, mode);
        ptiCheckOSError(!fp);
    }

    ~CFile() {
        if(fp) {
            std::fclose(fp);
            fp = nullptr;
        }
    }

    CFile& operator= (CFile const&) = delete;

    CFile& operator= (CFile&& other) {
        fp = other.fp;
        other.fp = nullptr;
        return *this;
    }

    operator std::FILE* () {
        return fp;
    }

    CFile& fopen(char const* pathname, char const* mode) {
        assert(!fp);
        fp = std::fopen(pathname, mode);
        ptiCheckOSError(!fp);
        return *this;
    }

    CFile& freopen(const char* pathname, char const* mode) {
        fp = std::freopen(pathname, mode, fp);
        ptiCheckOSError(!fp);
        return *this;
    }

    CFile& fclose() {
        assert(fp);
        int io_result = std::fclose(fp);
        fp = nullptr;
        ptiCheckOSError(io_result != 0);
        return *this;
    }

    std::FILE* c_file() {
        return fp;
    }

};

}

#endif
