export interface DeepseekConfig {
  apiKey: string
  baseUrl: string
  model: string
}

const defaultConfig: DeepseekConfig = {
  apiKey: process.env.DOUBAO_API_KEY || "",
  baseUrl: "https://ark.cn-beijing.volces.com/api/v3",
  model: "doubao-1-5-lite-32k-250115",
}

export const deepseek = (config: Partial<DeepseekConfig> = {}) => {
  const fullConfig = { ...defaultConfig, ...config }

  return async (messages: any[]) => {
    const response = await fetch(`${fullConfig.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${fullConfig.apiKey}`,
      },
      body: JSON.stringify({
        model: fullConfig.model,
        messages,
        stream: true,
      }),
    })

    if (!response.ok) {
      throw new Error(`Deepseek API request failed: ${response.statusText}`)
    }

    // 从响应中获取文本流
    const stream = response.body
    return stream
  }
}