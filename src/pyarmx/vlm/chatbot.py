import base64
import json
import time
from rich import print
from openai import OpenAI
from pyarmx.vlm.config import ModelCfg

LLM_TIMEOUT = 60.0   # 默认3分钟超时

class ChatBot:
    def __init__(self, config: 'ModelCfg', silent: bool = False):
        """初始化 ChatBot 实例"""
        
        self.silent = silent
        self.model = config.model
        self.system_prompt = config.system_prompt

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=LLM_TIMEOUT
        )
        self.conversation = [{"role": "system", "content": self.system_prompt}]

    # ===================== 工具函数 =====================
    def encode_img_path(self, image_path: str) -> str:
        """将图像编码为 base64 字符串"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def encode_img_cv2(self, img) -> str:
        """将 cv2 图片（numpy 数组）编码为 base64 字符串 """
        import cv2
        _, encoded_img = cv2.imencode('.jpg', img)
        return base64.b64encode(encoded_img.tobytes()).decode('utf-8')

    def extract_json(self, response_text: str) -> str | None:
        """从文本中提取 JSON 内容"""
        try:
            json.loads(response_text)
            return response_text
        except json.JSONDecodeError:
            pass

        first, last = response_text.find("{"), response_text.rfind("}")
        if first != -1 and last != -1:
            json_str = response_text[first : last + 1]
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        return None
    
    def json_loads(self, response_text: str) -> dict|None:
        """ 从响应文本中提取 JSON 字符串并加载为字典 """
        json_str = self.extract_json(response_text)
        if json_str is not None:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                print("解析提取的JSON字符串失败")
                return None
        return None

    # ===================== 公共辅助函数 =====================
    def _build_message_content(self, text: str, img_base64: str | None):
        """ 如果需要图片, 则构造 message 内容为 [text, image_url] """
        if img_base64:
            return [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            ]
        return text

    def _build_messages(self, content, use_history: bool):
        """ 如果需要历史对话, 则构造消息列表为 [system_prompt, user_input, assistant_output] """
        if use_history:
            msgs = self.conversation.copy()
            msgs.append({"role": "user", "content": content})
        else:
            msgs = [self.conversation[0], {"role": "user", "content": content}]
        return msgs

    def _build_api_params(self, messages, json_mode: bool):
        """ 统一构造 API 调用参数 """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "stream": True,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        return params

    def _stream_response(self, response):
        """统一处理流式输出"""
        if not self.silent:
            print(f"\n模型: {self.model} 流式输出响应: ", flush=True)
        
        parts = []
        first_token_time = None
        last_token_time = time.time()
        
        for chunk in response:
            # 无 token 超时保护 
            now = time.time()
            if now - last_token_time > LLM_TIMEOUT:
                raise TimeoutError("LLM streaming 长时间无响应")

            # 没有结果时重试获取结果
            if not chunk.choices:
                continue

            # 从
            content = chunk.choices[0].delta.content or ""
            if first_token_time is None and content:
                first_token_time = time.time()
            parts.append(content)
            
            if not self.silent:  # 输出流式结果
                print(content, end="", flush=True)
        
        if not self.silent:
            print("\n")  # 流式输出完成后换行

        return "".join(parts), first_token_time

    # ===================== 核心功能函数 =====================
    def chat(self, user_input: str, img_base64: str | None = None, json_mode=False, use_history=False) -> str | None:
        """普通聊天接口"""
        msg_content = self._build_message_content(user_input, img_base64)
        messages = self._build_messages(msg_content, use_history)
        params = self._build_api_params(messages, json_mode)

        response = self.client.chat.completions.create(**params)
        text, _ = self._stream_response(response)

        if use_history:
            self.conversation += [
                {"role": "user", "content": msg_content},
                {"role": "assistant", "content": text},
            ]
            
        if json_mode:
            return self.extract_json(text)

        return text 
    
    def chat_with_messages(self, messages: list[dict], json_mode: bool = False) -> str | None:
        params = self._build_api_params(messages, json_mode)
        response = self.client.chat.completions.create(**params)
        text, _ = self._stream_response(response)

        if json_mode:
            return self.extract_json(text)

        return text 


    def clear_history(self):
        """清除对话历史"""
        self.conversation = [self.conversation[0]]



if __name__ == "__main__":
    from pyarmx.vlm.config import configs, ModelID

    bot = ChatBot(configs[ModelID.LMS])
    text = bot.chat("你好")

    print(text)

#     text = """```jsonadasdasd
# {
#     "passed": "pass",
#     "comment": "画面仅1人主体且全身基本可见，运动为舞蹈动作，无多人/镜头异常/画质问题，符合采集标准。",
#     "description": "一名女性在花丛前做舞蹈动作"
# }asdasd
# ```
# """
#     json_str = bot.extract_json(text)

#     print(json_str)