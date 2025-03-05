from volcenginesdkarkruntime import Ark
import os
from typing import List, Dict, Any
from fastapi import Response
from fastapi.responses import StreamingResponse
import json

class DoubaoConfig:
    def __init__(self, model: str = "doubao-1-5-lite-32k-250115"):
        self.model = model

def doubao(config: DoubaoConfig = DoubaoConfig()):
    client = Ark(api_key=os.environ.get("ARK_API_KEY"))
    
    async def generate(messages: List[Dict[str, Any]]):
        try:
            stream = client.chat.completions.create(
                model=config.model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=1000
            )

            async def event_generator():
                for chunk in stream:
                    if chunk.choices:
                        # 构造与原始格式相同的响应
                        response_data = {
                            "choices": [{
                                "delta": {"content": chunk.choices[0].delta.content}
                            }]
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )

        except Exception as e:
            print(f"API调用错误: {str(e)}")
            raise e

    return generate