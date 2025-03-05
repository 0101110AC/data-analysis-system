from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from lib.doubao_1_5_lite_32k import doubao, DoubaoConfig
import os
import json

app = FastAPI()

# 添加CORS中间件，允许前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat = doubao(DoubaoConfig())

# 如果有静态文件，可以挂载静态文件目录
# app.mount("/static", StaticFiles(directory="static"), name="static")

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
            </style>
        </head>
        <body>
            <div class="sidebar">
                <input type="text" class="search-box" placeholder="搜索功能...">
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
                <div class="chat-area" id="chat-container"></div>
                <div class="input-area">
                    <input type="file" id="file-upload" style="display: none;">
                    <button onclick="document.getElementById('file-upload').click()">选择文件</button>
                    <input type="text" id="message-input" placeholder="输入您的问题或指令...">
                    <button onclick="sendMessage()">发送</button>
                </div>
            </div>

            <script>
                const chatContainer = document.getElementById('chat-container');
                const messageInput = document.getElementById('message-input');
                const fileUpload = document.getElementById('file-upload');
                
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
                        // 发送请求
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(messages)
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
                        console.error('请求错误:', error);
                        addMessage('assistant', '抱歉，发生了错误，请稍后再试。');
                    }
                }
                
                // 处理文件上传
                fileUpload.addEventListener('change', function(e) {
                    if (this.files && this.files[0]) {
                        const fileName = this.files[0].name;
                        addMessage('user', `上传文件: ${fileName}`);
                        // 这里可以添加文件处理逻辑
                        addMessage('assistant', `已接收文件: ${fileName}，请问您想要如何分析这些数据？`);
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

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    # 解析请求体
    messages = await request.json()
    return await chat(messages)