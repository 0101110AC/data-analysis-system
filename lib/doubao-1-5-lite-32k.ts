export interface DoubaoConfig { // 修改接口名称
  accessKey: string
  secretKey: string
  baseUrl: string
  model: string
}

const defaultConfig: DoubaoConfig = {
  accessKey: process.env.ARK_API_KEY || "",
  secretKey: "",
  baseUrl: "https://ark.cn-beijing.volces.com/api/v3",
  model: "doubao-1-5-lite-32k-250115",
}

export const doubao = (config: Partial<DoubaoConfig> = {}) => { // 修改函数名称
  const fullConfig = { ...defaultConfig, ...config }

  return async (messages: any[]) => {
    console.log("Sending request to DouBao API with config:", fullConfig); // 添加日志

    const response = await fetch(`${fullConfig.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${fullConfig.accessKey}`,
      },
      body: JSON.stringify({
        model: fullConfig.model,
        messages,
        stream: true,
        temperature: 0.7,
        max_tokens: 1000
      }),
    })

    if (!response.ok) {
      console.error(`DouBao API request failed: ${response.statusText}`); // 添加错误日志
      throw new Error(`DouBao API request failed: ${response.statusText}`)
    }

    console.log("Received response from DouBao API"); // 添加日志

    // 从响应中获取文本流并包装成NextResponse
    const reader = response.body?.getReader()
    const decoder = new TextDecoder()

    return new Response(new ReadableStream({
      async start(controller) {
        try {
          while (true) {
            const { done, value } = await reader!.read()
            if (done) break

            const chunk = decoder.decode(value)
            const lines = chunk.split('\n')

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6)
                if (data === '[DONE]') {
                  controller.close()
                  return
                }
                try {
                  const json = JSON.parse(data)
                  const text = json.choices[0]?.delta?.content || ''
                  if (text) {
                    controller.enqueue(`data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\n`)
                  }
                } catch (e) {
                  console.error('Error parsing JSON:', e)
                }
              }
            }
          }
        } catch (e) {
          controller.error(e)
        }
      }
    }), {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      }
    })

  }
}