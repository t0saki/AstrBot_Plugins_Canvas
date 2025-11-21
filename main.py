import base64
import os
import uuid
import io
import aiohttp
import mimetypes
from PIL import Image as PILImage  # 必须引入 PIL 进行图片清洗

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, register

MODEL_NAME = "gemini-3-pro-image-preview"

# 按照 Open WebUI 的逻辑，全部设为 BLOCK_NONE 以防止静默拦截
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

@register("AstrBot_Plugins_Canvas", "长安某", "gemini画图工具", "1.6.0")
class GeminiImageGenerator(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.api_keys = self.config.get("gemini_api_keys", [])
        self.current_key_index = 0
        
        # 初始化临时目录
        plugin_dir = os.path.dirname(__file__)
        self.save_dir = os.path.join(plugin_dir, "temp_images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.api_base_url = self.config.get(
            "api_base_url", "https://generativelanguage.googleapis.com"
        )

        if not self.api_keys:
            logger.error("未配置 Gemini API 密钥")

    def _get_current_api_key(self):
        if not self.api_keys: return None
        return self.api_keys[self.current_key_index]

    def _switch_next_api_key(self):
        if not self.api_keys: return
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

    # --- 核心：Open WebUI 同款图片清洗逻辑 ---
    def _process_image_for_api(self, image_path):
        """
        对图片进行标准化处理：
        1. 尺寸限制：最大 2048px (OpenWebUI 默认值)
        2. 颜色修正：移除 Alpha 通道，转 RGB
        3. 格式统一：转 JPEG (兼容性最好)
        """
        try:
            with PILImage.open(image_path) as img:
                # 1. 处理颜色模式 (防止透明底导致的伪影)
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    background = PILImage.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # 2. 尺寸限制 (限制最大边长 2048)
                max_dimension = 2048
                width, height = img.size
                if width > max_dimension or height > max_dimension:
                    ratio = min(max_dimension / width, max_dimension / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = img.resize(new_size, PILImage.Resampling.LANCZOS)
                    logger.info(f"图片优化: {width}x{height} -> {new_size}")

                # 3. 统一输出为 JPEG
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=90, optimize=True)
                return "image/jpeg", base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        except Exception as e:
            logger.error(f"图片优化失败，使用原始数据: {e}")
            # 兜底：如果处理失败，直接读原文件
            mime_type, _ = mimetypes.guess_type(image_path)
            with open(image_path, "rb") as f:
                return mime_type or "image/png", base64.b64encode(f.read()).decode('utf-8')

    # --- 指令处理 ---
    
    # 【修复】将 prompt 参数重命名为 msg，并给默认值，避免与框架参数注入冲突
    @filter.command("生成图片", alias={"文生图"})
    async def generate_image(self, event: AstrMessageEvent, msg: str = ""):
        if not self.api_keys:
            yield event.plain_result("请配置API Key")
            return
        
        # 如果 msg 为空，尝试获取事件的纯文本
        prompt = msg.strip()
        if not prompt:
            yield event.plain_result("请输入描述，例如：/生成图片 一只猫")
            return

        yield event.plain_result("正在生成...")
        
        # 【修复】使用 async for 遍历生成器
        async for result in self._execute_gemini_request(event, prompt, None):
            yield result

    # 【修复】将 prompt 参数重命名为 msg，并给默认值
    @filter.command("编辑图片", alias={"图编辑"})
    async def edit_image(self, event: AstrMessageEvent, msg: str = ""):
        if not self.api_keys:
            yield event.plain_result("请配置API Key")
            return
        
        image_path = await self._extract_image_from_reply(event)
        if not image_path:
            yield event.plain_result("请先引用(回复)一张图片")
            return

        prompt = msg.strip()
        # 编辑图片允许 prompt 为空（虽然通常需要指令，但有时只是风格化）
        if not prompt:
            prompt = "Enhance this image" # 默认指令

        yield event.plain_result("正在编辑...")
        
        # 【修复】使用 async for 遍历生成器
        async for result in self._execute_gemini_request(event, prompt, image_path):
            yield result

    # --- 统一执行逻辑 ---
    async def _execute_gemini_request(self, event, prompt, image_path):
        save_path = None
        image_data = None

        # 重试循环
        for _ in range(len(self.api_keys)):
            api_key = self._get_current_api_key()
            try:
                image_data = await self._call_api_naked(prompt, image_path, api_key)
                break # 成功则跳出
            except Exception as e:
                logger.error(f"Key {self.current_key_index} 失败: {e}")
                self._switch_next_api_key()

        if not image_data:
            yield event.plain_result("生成失败，请检查后台日志")
            return

        # 保存并发送
        try:
            file_name = f"{uuid.uuid4()}.png"
            save_path = os.path.join(self.save_dir, file_name)
            with open(save_path, "wb") as f:
                f.write(image_data)
            
            yield event.chain_result([Image.fromFileSystem(save_path)])
            logger.info(f"图片发送成功: {save_path}")
        except Exception as e:
            logger.error(f"保存/发送失败: {e}")
            yield event.plain_result(f"发送失败: {e}")
        finally:
            # 清理
            if image_path and os.path.exists(image_path):
                try: os.remove(image_path)
                except: pass
            if save_path and os.path.exists(save_path):
                try: os.remove(save_path)
                except: pass

    # --- API 调用 (完全模拟 Open WebUI / Curl) ---
    async def _call_api_naked(self, prompt, image_path, api_key):
        base_url = self.api_base_url.strip().rstrip("/")
        if not base_url.startswith("https://"):
            base_url = f"https://{base_url}"
        
        endpoint = f"{base_url}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        parts = []
        
        # 1. 添加文本 (Open WebUI 顺序：Text)
        parts.append({"text": prompt})

        # 2. 添加图片 (Open WebUI 顺序：Images appended after text)
        if image_path:
            mime_type, b64_data = self._process_image_for_api(image_path)
            parts.append({"inlineData": {"mimeType": mime_type, "data": b64_data}})

        # 3. 构造 Payload
        # 注意：这里完全不传 generationConfig，模拟 Curl 的默认行为
        payload = {
            "contents": [
                {"role": "user", "parts": parts}
            ],
            "safetySettings": SAFETY_SETTINGS
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url=endpoint, json=payload, headers=headers) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"API Error {response.status}: {text}")
                
                data = await response.json()

        # 解析返回
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            
            # 检查是否被拦截
            finish_reason = candidate.get("finishReason")
            if finish_reason == "SAFETY":
                raise Exception("图片生成因安全策略被拦截")
            
            # 提取图片
            content_parts = candidate.get("content", {}).get("parts", [])
            for part in content_parts:
                if "inlineData" in part:
                    return base64.b64decode(part["inlineData"]["data"])
        
        raise Exception("API未返回图片数据 (可能是生成了纯文本回复)")

    # --- 辅助：从引用中提取图片 ---
    async def _extract_image_from_reply(self, event: AstrMessageEvent):
        try:
            message_components = event.message_obj.message
            for comp in message_components:
                if isinstance(comp, Comp.Reply):
                    for quoted_comp in comp.chain:
                        if isinstance(quoted_comp, Comp.Image):
                            return await quoted_comp.convert_to_file_path()
        except Exception as e:
            logger.error(f"提取图片失败: {e}")
        return None

    async def terminate(self):
        """插件卸载清理"""
        if os.path.exists(self.save_dir):
            try:
                import shutil
                shutil.rmtree(self.save_dir)
            except: pass
