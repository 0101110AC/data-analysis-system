import { streamText } from "ai"
import { openai } from "@ai-sdk/openai"
import type { NextRequest } from "next/server"

export async function POST(req: NextRequest) {
  const { messages, file } = await req.json()

  let fileContent = ""
  if (file) {
    // TODO: Implement file processing logic
    fileContent = `Processed file: ${file.name}`
  }

  const result = streamText({
    model: openai("gpt-4o"),
    messages: [
      ...messages,
      {
        role: "system",
        content: `You are a data analysis assistant. ${fileContent ? `The user has uploaded a file: ${fileContent}` : ""}`,
      },
    ],
  })

  return result.toDataStreamResponse()
}

