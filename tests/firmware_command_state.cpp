#include <assert.h>
#include <string>
#include "../firmware/visiondrive_receiver/command_state.h"

const char *token = "0123456789abcdef0123456789abcdef";
std::string packet(unsigned sequence, const char *action, unsigned ttl=350,
                   const char *session="0123456789abcdef") {
    return std::string("VD1 ")+token+" "+session+" "+std::to_string(sequence)+" "+action+" "+std::to_string(ttl);
}

int main() {
    using namespace visiondrive;
    CommandState state(token);
    assert(state.action == Action::Stop);
    assert(!state.accept(packet(0,"FORWARD").c_str(),1,100));
    assert(state.accept(packet(0,"STOP").c_str(),1,100));
    assert(state.accept(packet(1,"FORWARD").c_str(),1,110));
    assert(state.action == Action::Forward);
    assert(!state.accept(packet(1,"LEFT").c_str(),1,120));
    assert(!state.accept(packet(0,"STOP").c_str(),1,120));
    assert(!state.accept(packet(2,"REVERSE").c_str(),1,120));
    assert(!state.accept(packet(2,"RIGHT",99).c_str(),1,120));
    assert(!state.accept(packet(2,"RIGHT",1001).c_str(),1,120));
    assert(!state.accept((packet(2,"RIGHT")+" extra").c_str(),1,120));
    assert(!state.accept("VD1 wrong_token 0123456789abcdef 2 RIGHT 350",1,120));
    assert(!state.accept((std::string("VD1 ")+token+" 0123456789abcdef -1 RIGHT 350").c_str(),1,120));
    assert(!state.accept((std::string("VD1 ")+token+" 0123456789abcdef 4294967296 RIGHT 350").c_str(),1,120));
    assert(!state.accept(packet(2,"RIGHT").c_str(),2,120));
    assert(!state.accept(packet(2,"RIGHT",350,"aaaaaaaaaaaaaaaa").c_str(),1,120));
    state.tick(459); assert(state.action == Action::Forward);
    state.tick(460); assert(state.action == Action::Stop);
    assert(!state.accept(packet(2,"RIGHT").c_str(),1,461));
    assert(state.accept(packet(3,"STOP").c_str(),1,462));
    assert(state.accept(packet(4,"RIGHT").c_str(),1,463));
    state.disconnect(); assert(state.action == Action::Stop);
    assert(!state.accept(packet(5,"FORWARD").c_str(),1,464));
    assert(state.accept(packet(0,"STOP",350,"aaaaaaaaaaaaaaaa").c_str(),2,465));
    assert(state.accept(packet(1,"LEFT",350,"aaaaaaaaaaaaaaaa").c_str(),2,466));
    assert(state.action == Action::Left);
    // Unsigned elapsed time must work across ESP32 millis() wraparound.
    CommandState wrap(token);
    assert(wrap.accept(packet(0,"STOP").c_str(),1,UINT32_MAX-100));
    assert(wrap.accept(packet(1,"FORWARD").c_str(),1,UINT32_MAX-90));
    wrap.tick(100); assert(wrap.action == Action::Forward);
    wrap.tick(259); assert(wrap.action == Action::Stop);
    puts("ESP32 protocol and watchdog checks passed");
}
