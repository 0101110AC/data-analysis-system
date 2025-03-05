// 基于Python后端API的聊天接口实现

export interface DoubaoConfig {
  model: string
}

const defaultConfig: DoubaoConfig = {
  model: "doubao-1-5-lite-32k-250115",
}

export const doubao = (config: Partial<DoubaoConfig> = {}) => {
  const fullConfig = { ...defaultConfig, ...config }

  return async (messages: any[]): Promise<Response> => {
    try {
      // 请求体
      const requestBody = {
        messages,
      };
      
      console.log("请求参数:", requestBody);
      
      // 构建URL - 使用本地Python FastAPI后端
      // 根据环境确定API URL
      const apiUrl = process.env.NODE_ENV === 'production' 
        ? '/api/chat'  // 生产环境使用相对路径
        : 'http://localhost:8000/api/chat';  // 开发环境使用完整URL
      
      // 设置请求头
      const headers: Record<string, string> = {
        "Content-Type": "application/json"
      };
      
      const response = await fetch(apiUrl, {
        method: "POST",
        headers,
        body: JSON.stringify(requestBody),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`API错误: ${response.status}`, errorText);
        throw new Error(`API请求失败: ${response.status} - ${errorText}`);
      }
      
      console.log("收到API响应");
      
      return new Response(response.body, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive'
        }
      });
    } catch (error) {
      console.error("API调用错误:", error);
      throw error;
    }
  }
}