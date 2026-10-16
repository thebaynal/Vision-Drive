#pragma once
// Pure C++ protocol/watchdog state: also exercised by host-side tests.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace visiondrive {
enum class Action { Stop, Forward, Left, Right };
inline const char *name(Action action) {
    switch (action) {
    case Action::Forward: return "FORWARD";
    case Action::Left: return "LEFT";
    case Action::Right: return "RIGHT";
    default: return "STOP";
    }
}

class CommandState {
public:
    explicit CommandState(const char *secret) : secret_(secret) {}
    Action action = Action::Stop;
    char session[17] = {};
    uint32_t sequence = 0;

    void tick(uint32_t now) {
        if (connected_ && uint32_t(now - received_) >= ttl_) {
            action = Action::Stop;
            armed_ = false;
        }
    }

    bool accept(const char *packet, uint64_t sender, uint32_t now) {
        tick(now);
        char version[5], token[33], incoming[17], word[9], serialText[11], ttlText[5], trailing;
        if (sscanf(packet, "%4s %32s %16s %10s %8s %4s %c", version, token,
                   incoming, serialText, word, ttlText, &trailing) != 6) return false;
        if (strspn(serialText, "0123456789") != strlen(serialText) ||
            strspn(ttlText, "0123456789") != strlen(ttlText)) return false;
        unsigned long long serial = strtoull(serialText, nullptr, 10);
        unsigned long ttl = strtoul(ttlText, nullptr, 10);
        if (strcmp(version, "VD1") || strcmp(token, secret_) || strlen(incoming) != 16 ||
            strspn(incoming, "0123456789abcdef") != 16 || serial > UINT32_MAX ||
            ttl < 100 || ttl > 1000) return false;
        Action next;
        if (!strcmp(word, "STOP")) next = Action::Stop;
        else if (!strcmp(word, "FORWARD")) next = Action::Forward;
        else if (!strcmp(word, "LEFT")) next = Action::Left;
        else if (!strcmp(word, "RIGHT")) next = Action::Right;
        else return false;

        bool same = connected_ && sender == sender_ && !strcmp(incoming, session);
        if (same && serial <= sequence) return false;
        // A new laptop session can only take ownership while stopped, using
        // a STOP handshake. Expired sessions cannot restart motors directly.
        if (!same && (next != Action::Stop || action != Action::Stop)) return false;
        if (same && !armed_ && next != Action::Stop) return false;
        strcpy(session, incoming);
        sequence = uint32_t(serial);
        sender_ = sender;
        received_ = now;
        ttl_ = uint32_t(ttl);
        connected_ = true;
        armed_ = true;
        action = next;
        return true;
    }

    void disconnect() { action = Action::Stop; armed_ = false; }

private:
    const char *secret_;
    bool connected_ = false, armed_ = false;
    uint64_t sender_ = 0;
    uint32_t received_ = 0, ttl_ = 350;
};
} // namespace visiondrive
