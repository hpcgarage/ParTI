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

#ifndef PTI_UTILS_INCLUDED
#define PTI_UTILS_INCLUDED

#include <cstring>
#include <string>
#include <sstream>
#include <stdexcept>

namespace pti {

template <typename T>
inline void unused_param(T&&) {
}

template <typename T>
inline T ceil_div(T const num, T const deno) {
    return num ? (num - 1) / deno + 1 : 0;
}

template <typename T>
inline std::string array_to_string(T const array[], size_t length, std::string const& delim = ", ") {
    if(length == 0) {
        return std::string();
    }
    std::ostringstream result;
    result << array[0];
    for(size_t i = 1; i < length; ++i) {
        result << delim << array[i];
    }
    return result.str();
}

class StrToNumError : public std::runtime_error {
public:
    StrToNumError() : std::runtime_error("Invalid number format") {}
};

template <typename Fn, typename ...Args>
static inline auto strtonum(Fn fn, const char *str, Args &&...args) -> decltype(fn(str, nullptr, std::forward<Args>(args)...)) {
    if(str[0] != '\0') {
        char *endptr;
        auto result = fn(str, &endptr, std::forward<Args>(args)...);
        if(endptr == &str[std::strlen(str)]) {
            return result;
        } else {
            throw StrToNumError();
        }
    } else {
        throw StrToNumError();
    }
}

static inline std::string size_to_string(size_t size) {
    std::ostringstream result;
    result.precision(1);
    if(size >= (1 << 30)) {
        result << std::fixed << (size / double(1 << 30)) << " GiB";
    } else if(size >= (1 << 20)) {
        result << std::fixed << (size / double(1 << 20)) << " MiB";
    } else if(size >= (1 << 10)) {
        result << std::fixed << (size / double(1 << 10)) << " KiB";
    } else if(size != 1) {
        result << size << " bytes";
    } else {
        result << "1 byte";
    }
    return result.str();
}

}

#ifdef PATCH_STD_TO_STRING_NOT_FOUND

namespace std {

template <typename T>
std::string to_string(T v) {
    std::ostringstream result;
    result << v;
    return result.str();
}

}

#endif

#endif
