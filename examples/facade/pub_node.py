import time
from rich import print as rprint
from rose.core import Publisher
from rose.message import ChatMsg

def main() -> None:
    addr = "tcp://127.0.0.1:5555"
    pub = Publisher(addr)
    rprint(f"[发布者] 已绑定 {addr}，开始广播...")
    
    counter = 0
    while True:
        msg = ChatMsg(content="当前状态正常", id=counter)
        pub.publish("status", msg)
        rprint(f"[Pub] 已发布: {msg}")
        counter += 1
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rprint("\n退出")