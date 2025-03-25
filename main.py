from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from lib.doubao_1_5_pro_32k import doubao, DoubaoConfig
from lib.ml_agent import ml_agent, MLAgentConfig
import os
import json
import shutil
from pathlib import Path

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # 允许的源
    allow_credentials=True,
    allow_methods=["*"],  # 允许的HTTP方法
    allow_headers=["*"],  # 允许的HTTP头
)

chat = doubao(DoubaoConfig())
# 使用本地部署的deepseek-r1:1.5b模型
ml_chat = ml_agent(MLAgentConfig(model="deepseek-r1:1.5b", use_local=True, local_url="http://localhost:11434"))

# 如果有静态文件，可以挂载静态文件目录
# app.mount("/static", StaticFiles(directory="static"), name="static")

# 添加/chat/ollama端点实现
@app.post("/chat/ollama")
async def chat_ollama(request: Request):
    # 解析请求体
    data = await request.json()
    messages = data if isinstance(data, list) else data.get("messages", [])
    
    # 使用机器学习代理处理请求
    return await ml_chat(messages)

@app.get("/")
async def root():
    return HTMLResponse("""
    <html>
        <head>
            <title>数据分析助手</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 0; 
                    display: flex;
                    height: 100vh;
                }
                .sidebar {
                    width: 200px;
                    background-color: #f5f5f5;
                    padding: 20px;
                    border-right: 1px solid #ddd;
                    overflow-y: auto;
                }
                .main-content {
                    flex: 1;
                    padding: 20px;
                    display: flex;
                    flex-direction: column;
                }
                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }
                .chat-area {
                    flex: 1;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 20px;
                    margin-bottom: 20px;
                    overflow-y: auto;
                }
                .input-area {
                    display: flex;
                    gap: 10px;
                }
                input[type="text"] {
                    flex: 1;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                button {
                    padding: 10px 20px;
                    background-color: #000;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                .search-box {
                    width: 100%;
                    padding: 8px;
                    margin-bottom: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .category {
                    margin-bottom: 10px;
                }
                .category-title {
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                .method-list {
                    list-style-type: none;
                    padding-left: 10px;
                    margin: 0;
                }
                .method-item {
                    padding: 5px 0;
                    cursor: pointer;
                }
                .method-item:hover {
                    text-decoration: underline;
                }
                .language-switcher {
                    display: flex;
                    align-items: center;
                }
                .agent-selector {
                    margin-bottom: 10px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                select {
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: white;
                }
            </style>
        </head>
        <body>
            <div class="sidebar">
                <label for="search-box">搜索：</label>
                <input type="text" id="search-box" class="search-box" placeholder="搜索功能..." title="搜索功能">
                <div class="category">
                    <div class="category-title">类别</div>
                    <ul class="method-list">
                        <li class="method-item">全部</li>
                        <li class="method-item">图表性</li>
                        <li class="method-item">变量性</li>
                        <li class="method-item">相关性</li>
                    </ul>
                </div>
                <div class="category">
                    <div class="category-title">方法</div>
                    <ul class="method-list">
                        <li class="method-item">Linear Regression</li>
                        <li class="method-item">Logistic Regression</li>
                        <li class="method-item">Decision Trees</li>
                        <li class="method-item">Random Forests</li>
                        <li class="method-item">K-Means Clustering</li>
                        <li class="method-item">Hierarchical Clustering</li>
                        <li class="method-item">Principal Component Analysis</li>
                    </ul>
                </div>
            </div>
            <div class="main-content">
                <div class="header">
                    <h1>数据分析助手</h1>
                    <div class="language-switcher">
                        <span>Data Analysis Assistant</span>
                    </div>
                </div>
                <div class="agent-selector">
                    <label for="agent-select">选择智能体:</label>
                    <select id="agent-select">
                        <option value="default">火山引擎大模型</option>
                        <option value="ml-agent">机器学习专家（本地）</option>
                    </select>
                </div>
                <div class="chat-area" id="chat-container"></div>
                <div class="input-area">
                    <input type="file" id="file-upload" style="display: none;" title="上传文件">
                    <button onclick="document.getElementById('file-upload').click()">选择文件</button>
                    <input type="text" id="message-input" placeholder="输入您的问题或指令..." title="消息输入框">
                    <button onclick="sendMessage()">发送</button>
                </div>
            </div>

            <script>
                const chatContainer = document.getElementById('chat-container');
                const messageInput = document.getElementById('message-input');
                const fileUpload = document.getElementById('file-upload');
                const agentSelect = document.getElementById('agent-select');
                
                // 存储对话历史
                const messages = [
                    {role: "system", content: "你是一个数据分析助手，可以帮助用户分析数据、选择合适的分析方法、解释结果等。"}
                ];
                
                // 添加消息到界面
                function addMessage(role, content) {
                    const messageDiv = document.createElement('div');
                    messageDiv.style.marginBottom = '10px';
                    messageDiv.style.padding = '8px';
                    messageDiv.style.borderRadius = '5px';
                    
                    if (role === 'user') {
                        messageDiv.style.backgroundColor = '#e6f7ff';
                        messageDiv.style.textAlign = 'right';
                    } else {
                        messageDiv.style.backgroundColor = '#f0f0f0';
                    }
                    
                    messageDiv.textContent = `${role === 'user' ? '我' : 'AI'}: ${content}`;
                    chatContainer.appendChild(messageDiv);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                
                // 发送消息
                async function sendMessage() {
                    const content = messageInput.value.trim();
                    if (!content) return;
                    
                    // 添加用户消息到界面
                    addMessage('user', content);
                    
                    // 添加到消息历史
                    messages.push({role: "user", content});
                    messageInput.value = '';
                    
                    try {
                        // 获取选择的智能体
                        const selectedAgent = agentSelect.value;
                        
                        // 发送请求
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                messages: messages,
                                body: {
                                    agent: selectedAgent
                                }
                            })
                        });
                        
                        if (response.ok) {
                            const reader = response.body.getReader();
                            const decoder = new TextDecoder();
                            let aiResponse = '';
                            
                            // 处理流式响应
                            while (true) {
                                const {done, value} = await reader.read();
                                if (done) break;
                                
                                const chunk = decoder.decode(value);
                                const lines = chunk.split('\\n');
                                
                                for (const line of lines) {
                                    if (!line.trim() || line === 'data: [DONE]') continue;
                                    
                                    if (line.startsWith('data: ')) {
                                        try {
                                            const jsonData = JSON.parse(line.slice(6));
                                            // 检查是否有错误响应
                                            if (jsonData.error) {
                                                console.error('API错误:', jsonData.message);
                                                addMessage('assistant', `抱歉，发生了错误: ${jsonData.message}`);
                                                continue;
                                            }
                                            
                                            if (jsonData.choices && jsonData.choices[0]?.delta?.content) {
                                                const content = jsonData.choices[0].delta.content;
                                                aiResponse += content;
                                                // 实时更新AI回复
                                                const aiMessageDiv = document.querySelector('#ai-response');
                                                if (aiMessageDiv) {
                                                    aiMessageDiv.textContent = `AI: ${aiResponse}`;
                                                } else {
                                                    const newDiv = document.createElement('div');
                                                    newDiv.id = 'ai-response';
                                                    newDiv.style.backgroundColor = '#f0f0f0';
                                                    newDiv.style.padding = '8px';
                                                    newDiv.style.borderRadius = '5px';
                                                    newDiv.style.marginBottom = '10px';
                                                    newDiv.textContent = `AI: ${aiResponse}`;
                                                    chatContainer.appendChild(newDiv);
                                                }
                                            }
                                        } catch (e) {
                                            console.error('解析响应数据错误:', e);
                                        }
                                    }
                                }
                            }
                            
                            // 添加完整的AI回复到消息历史
                            if (aiResponse) {
                                messages.push({role: "assistant", content: aiResponse});
                                // 移除临时的响应div
                                const aiMessageDiv = document.querySelector('#ai-response');
                                if (aiMessageDiv) {
                                    aiMessageDiv.id = '';
                                }
                            }
                        } else {
                            console.error('请求失败:', response.status);
                            addMessage('assistant', '抱歉，发生了错误，请稍后再试。');
                        }
                    } catch (error) {
                        // 改进错误处理，避免尝试读取undefined对象的属性
                        console.error('请求错误:', error);
                        let errorMessage = '抱歉，发生了错误，请稍后再试。';
                        // 安全地检查error对象
                        if (error && typeof error === 'object') {
                            errorMessage = error.message || errorMessage;
                        }
                        addMessage('assistant', errorMessage);
                    }
                }
                
                // 处理文件上传
                fileUpload.addEventListener('change', function(e) {
                    if (this.files && this.files[0]) {
                        const fileName = this.files[0].name;
                        addMessage('user', `上传文件: ${fileName}`);
                        
                        // 创建FormData对象上传文件
                        const formData = new FormData();
                        formData.append('file', this.files[0]);
                        
                        // 上传文件到服务器
                        fetch('/api/upload', {
                            method: 'POST',
                            body: formData
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                // 文件上传成功，添加文件信息到下一次请求
                                const fileInfo = {
                                    name: fileName,
                                    path: data.filePath
                                };
                                
                                // 存储文件信息以便在发送消息时使用
                                window.uploadedFile = fileInfo;
                                
                                addMessage('assistant', `已接收文件: ${fileName}，请问您想要如何分析这些数据？`);
                            } else {
                                addMessage('assistant', '文件上传失败，请重试。');
                            }
                        })
                        .catch(error => {
                            console.error('文件上传错误:', error);
                            addMessage('assistant', '文件上传失败，请重试。');
                        });
                    }
                });
                
                // 按Enter发送消息
                messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') sendMessage();
                });
                
                // 点击方法时的处理
                document.querySelectorAll('.method-item').forEach(item => {
                    item.addEventListener('click', function() {
                        const method = this.textContent;
                        messageInput.value = `请解释${method}方法是什么以及如何使用`;
                    });
                });
            </script>
        </body>
    </html>
    """)


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    # 确保上传目录存在
    upload_dir = Path("public/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存上传的文件
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return JSONResponse({
        "success": True,
        "fileName": file.filename,
        "filePath": str(file_path)
    })

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    # 解析请求体
    data = await request.json()
    messages = data if isinstance(data, list) else data.get("messages", [])
    
    # 检查是否有文件信息和智能体选择
    file_info = None
    agent_type = None
    if isinstance(data, dict) and "body" in data:
        if "file" in data["body"]:
            file_info = data["body"]["file"]
        if "agent" in data["body"]:
            agent_type = data["body"]["agent"]
    
    # 如果有文件信息，添加到系统消息中
    if file_info and isinstance(messages, list) and len(messages) > 0:
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                messages[i]["content"] += f"\n用户上传了文件: {file_info['name']}，请帮助分析这个文件。"
                break
        else:
            # 如果没有找到系统消息，添加一个
            messages.insert(0, {
                "role": "system", 
                "content": f"你是一个数据分析助手，可以帮助用户分析数据。用户上传了文件: {file_info['name']}，请帮助分析这个文件。"
            })
    
    # 根据选择的智能体类型调用不同的AI模型
    if agent_type == "ml-agent":
        return await ml_chat(messages)
    else:
        return await chat(messages)

@app.post("/api/ml_chat")
async def ml_chat_endpoint(request: Request):
    # 解析请求体
    data = await request.json()
    messages = data if isinstance(data, list) else data.get("messages", [])
    
    # 检查是否有文件信息
    file_info = None
    if isinstance(data, dict) and "body" in data and "file" in data["body"]:
        file_info = data["body"]["file"]
    
    # 如果有文件信息，添加到系统消息中
    if file_info and isinstance(messages, list) and len(messages) > 0:
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                messages[i]["content"] += f"\n用户上传了文件: {file_info['name']}，请帮助分析这个文件中的机器学习相关内容。"
                break
        else:
            # 如果没有找到系统消息，添加一个
            messages.insert(0, {
                "role": "system", 
                "content": f"你是一个机器学习专家，可以帮助用户解决机器学习问题。用户上传了文件: {file_info['name']}，请帮助分析这个文件中的机器学习相关内容。"
            })
    
    return await ml_chat(messages)