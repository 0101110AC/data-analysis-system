from typing import List, Dict, Any
from fastapi import Response
from fastapi.responses import StreamingResponse
import json
import asyncio
import aiohttp
from lib.doubao_1_5_pro_32k import doubao, DoubaoConfig
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

class MLAgentConfig:
    def __init__(self, model: str = "deepseek-r1:1.5b", temperature: float = 0.7, max_tokens: int = 1000, use_local: bool = False, local_url: str = "http://localhost:11434"):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local = use_local
        self.local_url = local_url

# 创建一个专注于机器学习和深度学习的系统提示
ML_SYSTEM_PROMPT = """
你是一个专业的机器学习和深度学习专家，精通各种算法及其应用。你的知识库包括但不限于：

1. 监督学习算法：线性回归、逻辑回归、决策树、随机森林、支持向量机、神经网络等
2. 无监督学习算法：K均值聚类、层次聚类、DBSCAN、主成分分析(PCA)、t-SNE等
3. 深度学习架构：CNN、RNN、LSTM、GRU、Transformer、注意力机制等
4. 强化学习：Q-learning、DQN、策略梯度、Actor-Critic等
5. 模型评估与优化：交叉验证、网格搜索、贝叶斯优化、学习曲线分析等
6. 特征工程：特征选择、降维、归一化、标准化等
7. 数据预处理：缺失值处理、异常值检测、数据增强等
8. 模型解释性：SHAP值、LIME、特征重要性等

你的回答应该准确、专业，并且能够根据用户的问题提供适当的代码示例、算法选择建议或实现思路。
当用户询问特定算法或技术时，你应该能够解释其原理、优缺点、适用场景以及实现方法。

请记住，你的目标是帮助用户理解和应用机器学习与深度学习技术，解决实际问题。
"""

def ml_agent(config: MLAgentConfig = MLAgentConfig()):
    """
    创建一个专注于机器学习和深度学习的智能体
    """
    # 如果不使用本地模型，则使用豆包API
    if not config.use_local:
        doubao_client = doubao(DoubaoConfig(model=config.model))
    
    async def generate(messages: List[Dict[str, Any]]):
        try:
            # 确保系统消息包含ML专家提示
            has_system_message = False
            for message in messages:
                if message.get("role") == "system":
                    has_system_message = True
                    # 如果已有系统消息，确保它包含ML专家提示
                    if ML_SYSTEM_PROMPT not in message["content"]:
                        message["content"] = f"{message['content']}\n\n{ML_SYSTEM_PROMPT}"
                    break
            
            # 如果没有系统消息，添加一个
            if not has_system_message:
                messages.insert(0, {
                    "role": "system",
                    "content": ML_SYSTEM_PROMPT
                })
            
            # 根据配置选择使用本地模型或豆包API
            if config.use_local:
                # 使用本地部署的模型API
                return await call_local_model(config, messages)
            else:
                # 使用豆包API处理消息
                return await doubao_client(messages)
            
        except Exception as e:
            print(f"ML Agent API调用错误: {str(e)}")
            error_response = {
                "error": True,
                "message": f"ML Agent API调用错误: {str(e)}"
            }
            return Response(
                content=json.dumps(error_response),
                media_type="application/json",
                status_code=500
            )
    
    async def call_local_model(config: MLAgentConfig, messages: List[Dict[str, Any]]):
        """
        使用LangChain框架调用本地部署的模型API
        """
        try:
            # 使用LangChain的OllamaLLM初始化模型，启用流式处理
            llm = OllamaLLM(
                model=config.model,
                base_url=config.local_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                streaming=True
            )
            
            # 将消息列表转换为LangChain消息格式
            langchain_messages = []
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
            
            # 创建流式响应
            async def event_generator():
                try:
                    # 使用LangChain的流式处理
                    stream = llm.stream(langchain_messages)
                    
                    # 标记当前是思考过程还是正式回答
                    is_thinking = True
                    
                    for chunk in stream:
                        content = chunk if isinstance(chunk, str) else chunk.content
                        
                        # 检测是否从思考过程转换为正式回答
                        if is_thinking and content.strip().startswith("<answer>"):
                            is_thinking = False
                            # 移除<answer>标记
                            content = content.replace("<answer>", "", 1).lstrip()
                        
                        # 构造与前端期望格式一致的响应
                        response_data = {
                            "choices": [{
                                "delta": {"content": content}
                            }],
                            "response_type": "thinking" if is_thinking else "answer"
                        }
                        
                        # 转换为JSON并添加SSE格式
                        json_str = json.dumps(response_data, ensure_ascii=False)
                        yield f"data: {json_str}\n\n"
                    
                    # 发送完成标记
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    error_msg = f"LangChain流处理错误: {str(e)}"
                    print(error_msg)
                    yield f"data: {{\"error\": true, \"message\": \"{error_msg}\"}}\n\n"
            
            # 返回流式响应
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
            print(f"本地模型API调用错误: {str(e)}")
            error_response = {
                "error": True,
                "message": f"本地模型API调用错误: {str(e)}"
            }
            return Response(
                content=json.dumps(error_response),
                media_type="application/json",
                status_code=500
            )
    
    return generate