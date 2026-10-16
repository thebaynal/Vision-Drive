"""Bounded Wi-Fi UDP command link. The receiver owns its motor watchdog.

Wire format: VD1 token session sequence ACTION ttl_ms
Reply: VA1 session sequence ACTION
This is a local-network demo protocol, not encrypted/authenticated transport.
"""

import re
import secrets
import socket
import time
from .decision import Action


class ESPLink:
    def __init__(self, host, token, port=4210, ttl_ms=350, clock=time.monotonic):
        if not re.fullmatch(r"[0-9a-fA-F]{32}", token or ""):
            raise ValueError("VISIONDRIVE_ESP_TOKEN must contain 32 hexadecimal characters; see the Wi-Fi guide")
        if not 100 <= ttl_ms <= 1000 or not 1 <= port <= 65535:
            raise ValueError("Use a command TTL of 100-1000 ms and a valid UDP port")
        self.address = (socket.gethostbyname(host), port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("", 0))
        self.socket.setblocking(False)
        self.token, self.session, self.sequence = token, secrets.token_hex(8), 0
        self.ttl_ms, self.clock = ttl_ms, clock
        self.last_send = -float("inf")
        self.last_ack = -float("inf")
        self.last_ack_sequence = -1
        self.ready = False
        self.last_action = Action.STOP
        self.status = "ESP32: waiting for STOP acknowledgement"

    def _receive(self):
        for _ in range(32):
            try:
                payload, address = self.socket.recvfrom(128)
            except (BlockingIOError, ConnectionResetError):
                return
            if address != self.address:
                continue
            parts = payload.decode("ascii", errors="ignore").split()
            if len(parts) != 4 or parts[:2] != ["VA1", self.session] or not parts[2].isdecimal():
                continue
            sequence = int(parts[2])
            if sequence <= self.last_ack_sequence or sequence >= self.sequence or parts[3] not in Action._value2member_map_:
                continue
            self.last_ack_sequence = sequence
            self.last_ack = self.clock()
            # Initial session ownership must be acknowledged while stopped.
            if parts[3] == "STOP":
                self.ready = True

    def send(self, decision, stale_after_ms, force=False):
        now = self.clock()
        self._receive()
        if now-self.last_ack > 1.:
            self.ready = False
        fresh = (decision.frame_id >= 0 and 0 <= now-decision.captured_at < stale_after_ms/1000)
        action = decision.action if self.ready and fresh else Action.STOP
        if not force and now-self.last_send < .05:
            return
        # Never extend a movement command beyond the perception's expiry.
        remaining_ms = int((decision.captured_at + stale_after_ms/1000-now)*1000) if fresh else self.ttl_ms
        ttl = min(self.ttl_ms, remaining_ms)
        if ttl < 100:
            action, ttl = Action.STOP, self.ttl_ms
        message = f"VD1 {self.token} {self.session} {self.sequence} {action.value} {ttl}"
        self.socket.sendto(message.encode("ascii"), self.address)
        self.sequence += 1
        self.last_send = now
        self.last_action = action
        self.status = f"ESP32: {'linked' if self.ready else 'waiting for ACK'} / sent {action.value}"

    def close(self):
        # Redundant STOP datagrams; loss is also covered by receiver expiry.
        for _ in range(3):
            message = f"VD1 {self.token} {self.session} {self.sequence} STOP {self.ttl_ms}"
            try:
                self.socket.sendto(message.encode("ascii"), self.address)
            except OSError:
                break
            self.sequence += 1
        self.socket.close()
