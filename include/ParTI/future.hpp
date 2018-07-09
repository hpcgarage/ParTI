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

#ifndef PTI_FUTURE_INCLUDED
#define PTI_FUTURE_INCLUDED

#include <functional>
#include <list>
#include <memory>
#include <string>

namespace pti {

template <typename T>
class Future;

template <>
class Future<void> {

public:

    typedef std::function<void ()> procedure;
    typedef std::function<void (std::exception)> exception_handler;

private:

    struct named_procedure {
        procedure proc;
        std::string name;
        bool finished;
    };

public:

    Future() {}
    Future(const std::string &name, procedure proc);
    Future(procedure proc);
    virtual ~Future();

    Future &then(const std::string &name, procedure proc);
    Future &then(procedure proc);
    Future &then(Future *future);

    Future &onerror(exception_handler handler);

private:

    Future *prev = nullptr;
    Future *next = nullptr;
    Future *begin = this;
    Future *end = this;
    std::list<named_procedure> procedures;
    std::list<exception_handler> exception_handlers;

};

template <typename T>
class Future : public Future<void> {

public:

    Future() {}
    Future(procedure proc) : Future<void>(proc) {}
    ~Future() {}

    T data;

};

}

#endif
