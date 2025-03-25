from volcenginesdkarkruntime import Ark
import os
from typing import List, Dict, Any
from fastapi import Response
from fastapi.responses import StreamingResponse
import json

class DoubaoConfig:
    def __init__(self, model: str = "doubao-1-5-pro-32k-250115"):
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
                    try:
                        # 构造与AI库期望格式一致的响应
                        response_data = {"choices": [{}]}
                        
                        # 处理content类型的响应
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                            response_data["choices"][0]["delta"] = {"content": chunk.choices[0].delta.content or ""}
                        
                        # 处理tool_calls类型的响应
                        elif chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.tool_calls:
                            tool_calls_data = []
                            for tool_call in chunk.choices[0].delta.tool_calls:
                                tool_call_data = {
                                    "index": tool_call.index,
                                    "type": tool_call.type
                                }
                                
                                if tool_call.id:
                                    tool_call_data["id"] = tool_call.id
                                    
                                if tool_call.function:
                                    function_data = {}
                                    if tool_call.function.name:
                                        function_data["name"] = tool_call.function.name
                                    if tool_call.function.arguments:
                                        function_data["arguments"] = tool_call.function.arguments
                                    
                                    if function_data:
                                        tool_call_data["function"] = function_data
                                
                                tool_calls_data.append(tool_call_data)
                            
                            if tool_calls_data:
                                response_data["choices"][0]["delta"] = {"tool_calls": tool_calls_data}
                        
                        # 处理finish_reason
                        if chunk.choices and chunk.choices[0].finish_reason:
                            response_data["choices"][0]["finish_reason"] = chunk.choices[0].finish_reason
                        
                        # 只有当有实际内容时才发送数据
                        if "delta" in response_data["choices"][0] or "finish_reason" in response_data["choices"][0]:
                            try:
                                # 确保JSON格式正确，并添加正确的SSE格式
                                json_str = json.dumps(response_data, ensure_ascii=False)
                                # 使用正确的格式：data: 前缀，这是标准SSE格式
                                yield f"data: {json_str}\n\n"
                            except Exception as json_error:
                                print(f"JSON序列化错误: {str(json_error)}")
                                # 跳过无效的数据块
                                continue
                    except Exception as e:
                        print(f"生成响应数据错误: {str(e)}")
                        # 跳过无效的数据块
                        continue
                # 确保最后发送[DONE]标记，使用标准SSE格式
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream"
                }
            )

        except Exception as e:
            print(f"API调用错误: {str(e)}")
            # 返回错误响应而不是直接抛出异常
            error_response = {
                "error": True,
                "message": f"API调用错误: {str(e)}"
            }
            return Response(
                content=json.dumps(error_response),
                media_type="application/json",
                status_code=500
            )

    return generate