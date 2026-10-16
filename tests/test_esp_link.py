import socket
import time
from visiondrive.config import Config
from visiondrive.decision import Action, Controller
from visiondrive.esp_link import ESPLink

TOKEN = "0123456789abcdef0123456789abcdef"


def receive(server):
    payload, address = server.recvfrom(256)
    return payload.decode().split(), address


def acknowledge(server, fields, address):
    server.sendto(f"VA1 {fields[2]} {fields[3]} {fields[4]}".encode(), address)
    time.sleep(.005)


def test_handshake_stale_perception_disconnect_and_final_stop():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1",0))
    server.settimeout(1.)
    now = [100.]
    link = ESPLink("127.0.0.1",TOKEN,server.getsockname()[1],clock=lambda:now[0])
    ctl = Controller(Config(stable_frames=1))
    fresh = ctl.update([],0,100.,100.)
    try:
        link.send(fresh,1000,force=True)
        fields, address = receive(server)
        assert fields[4] == "STOP" # Movement cannot bypass the handshake.
        acknowledge(server,fields,address)
        link.send(fresh,1000,force=True)
        fields,address = receive(server)
        assert fields[4] == "FORWARD"
        acknowledge(server,fields,address)
        now[0] = 100.95
        link.send(fresh,1000,force=True)
        fields,address = receive(server)
        assert fields[4] == "STOP" # Remaining freshness below minimum packet TTL.
        now[0] = 102.
        new = ctl.update([],1,102.,102.)
        link.send(new,1000,force=True)
        fields,address = receive(server)
        assert fields[4] == "STOP" and not link.ready # ACK timeout.
        link.close()
        fields,_ = receive(server)
        assert fields[4] == "STOP"
    finally:
        if link.socket.fileno() >= 0: link.close()
        server.close()


def test_ack_cannot_be_replayed_to_keep_link_alive():
    server = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    server.bind(("127.0.0.1",0)); server.settimeout(1.)
    now = [100.]
    link = ESPLink("127.0.0.1",TOKEN,server.getsockname()[1],clock=lambda:now[0])
    fresh = Controller(Config(stable_frames=1)).update([],0,100.,100.)
    try:
        link.send(fresh,1000,force=True)
        original,address = receive(server)
        acknowledge(server,original,address)
        link.send(fresh,1000,force=True); receive(server)
        now[0] = 102.
        acknowledge(server,original,address)
        link.send(fresh,1000,force=True)
        fields,_ = receive(server)
        assert fields[4] == "STOP" and not link.ready
    finally:
        link.close(); server.close()
