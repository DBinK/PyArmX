from rich import print as rprint
from rose.core import Subscriber
from rose.message import ChatMsg

def main() -> None:
    addr = "tcp://127.0.0.1:5558"
    sub = Subscriber(addr, "status", ChatMsg)
    rprint(f"[订阅者] 已连接 {addr}，监听 status 话题...")
    
    while True:
        msg = sub.recv()
        rprint(f"[Sub] 收到数据: {msg}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rprint("\n退出")