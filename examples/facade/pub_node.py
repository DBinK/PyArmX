
import time
from rich import print as rprint
from rose.core import Publisher
from rose.message import ChatMsg

def main() -> None:
    addr = "tcp://127.0.0.1:5559"
    pub = Publisher(addr)
    rprint(f"[发布者] 已绑定 {addr}，开始广播...")
    
    counter = 0
    while True:
        try:
            user_input = input("请输入消息内容 (输入 'quit' 退出): ")
            
            if user_input.lower() == 'quit':
                rprint("[Pub] 用户退出")
                break            
            
            if user_input.lower() == '':
                rprint("input is empyt")
                continue

            if user_input.lower() == ']':
                user_input = "把红色方块放到胶带里"

            if user_input.lower() == '[':
                user_input = "[把红色方块放到胶带里]"
            
            msg = ChatMsg(content=user_input, id=counter)
            pub.publish("status", msg)
            rprint(f"[Pub] 已发布: {msg}\n")
            counter += 1
            
        except KeyboardInterrupt:
            rprint("\n[Pub] 程序被中断")
            break
        except EOFError:
            rprint("\n[Pub] 输入流结束")
            break

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        rprint(f"\n发生错误: {e}")