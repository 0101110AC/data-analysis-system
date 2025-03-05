import type { NextRequest } from "next/server"
import { doubao } from "@/lib/doubao-1-5-lite-32k"

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

    if (!messages || !Array.isArray(messages)) {
      return new Response(
        JSON.stringify({ error: "消息格式不正确" }),
        { 
          status: 400, 
          headers: { 
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*", // 添加CORS头
          } 
        }
      );
    }

    const result = await doubao({
      model: "doubao-1-5-lite-32k-250115",
    })(allMessages)

    // 添加CORS头
    const headers = new Headers(result.headers);
    headers.set('Access-Control-Allow-Origin', '*');

    return new Response(result.body, {
      status: result.status,
      headers
    });
  } catch (error) {
    console.error("详细错误信息:", error);
    return new Response(
      JSON.stringify({
        error: "处理请求时发生错误",
        details: error instanceof Error ? error.message : String(error)
      }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      }
    )
  }
}

