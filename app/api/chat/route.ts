import type { NextRequest } from "next/server"
import { deepseek } from "@/lib/deepseek"

export async function POST(req: NextRequest) {
  try {
    const { messages, file } = await req.json()

    let fileContent = ""
    if (file) {
      // TODO: Implement file processing logic
      fileContent = `Processed file: ${file.name}`
    }

    // 确保系统消息在最前面
    const systemMessage = {
      role: "system",
      content: "你是人工智能助手.",
    }

    const allMessages = [
      systemMessage,
      ...messages,
      fileContent ? { role: "user", content: fileContent } : null,
    ].filter(Boolean)

    const result = await deepseek({
      model: "doubao-1-5-lite-32k-250115",
    })(allMessages)

    return result
  } catch (error) {
    console.error("Chat API Error:", error)
    return new Response(
      JSON.stringify({
        error: "处理请求时发生错误",
      }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
  }
}

