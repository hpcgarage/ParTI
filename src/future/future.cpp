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

#include <ParTI/future.hpp>
#include <utility>

namespace pti {

Future<void>::Future(const std::string &name, procedure proc) {
    this->then(name, proc);
}

Future<void>::Future(procedure proc) {
    this->then(proc);
}

Future<void>::~Future() {
    if(this->prev != nullptr) {
        this->prev->next = nullptr;
        delete this->prev;
    }
    if(this->next != nullptr) {
        this->next->prev = nullptr;
        delete this->next;
    }
}

Future<void> &Future<void>::then(const std::string &name, procedure proc) {
    named_procedure named_proc;
    named_proc.proc = proc;
    named_proc.name = name;
    named_proc.finished = false;
    this->end->procedures.push_back(std::move(named_proc));
    return *this;
}

Future<void> &Future<void>::then(procedure proc) {
    named_procedure named_proc;
    named_proc.proc = proc;
    named_proc.finished = false;
    this->end->procedures.push_back(std::move(named_proc));
    return *this;
}

Future<void> &Future<void>::then(Future *future) {
    future->begin->prev = this;
    this->next = future->begin;
    this->end = future->end;
    return *this;
}

Future<void> &Future<void>::onerror(exception_handler handler) {
    this->exception_handlers.push_back(handler);
    return *this;
}

}
