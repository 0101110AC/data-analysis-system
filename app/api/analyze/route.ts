import { type NextRequest, NextResponse } from "next/server"
import { readFile } from "fs/promises"
import path from "path"

// Simple classification function to replace the natural library
function simpleClassify(text: string): string {
  const lowercaseText = text.toLowerCase()
  if (lowercaseText.includes("positive") || lowercaseText.includes("good") || lowercaseText.includes("great")) {
    return "positive"
  } else if (
    lowercaseText.includes("negative") ||
    lowercaseText.includes("bad") ||
    lowercaseText.includes("terrible")
  ) {
    return "negative"
  } else {
    return "neutral"
  }
}

export async function POST(req: NextRequest) {
  const { fileName, text } = await req.json()

  if (fileName) {
    const filePath = path.join(process.cwd(), "public", "uploads", fileName)
    const fileContent = await readFile(filePath, "utf-8")

    // Perform analysis on file content
    const classification = simpleClassify(fileContent)
    return NextResponse.json({ result: `File content classified as: ${classification}` })
  } else if (text) {
    // Perform analysis on text input
    const classification = simpleClassify(text)
    return NextResponse.json({ result: `Text classified as: ${classification}` })
  } else {
    return NextResponse.json({ error: "No file or text provided" }, { status: 400 })
  }
}

