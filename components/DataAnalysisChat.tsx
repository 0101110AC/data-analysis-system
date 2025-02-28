"use client"

import type React from "react"

import { useState } from "react"
import { useChat } from "ai/react"
import { useLanguage } from "@/contexts/LanguageContext"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"

export default function DataAnalysisChat() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: "/api/chat",
  })
  const [file, setFile] = useState<File | null>(null)
  const { translations } = useLanguage()

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0])
    }
  }

  const handleFormSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    let uploadedFileName = null
    
    if (file) {
      const formData = new FormData()
      formData.append("file", file)

      try {
        const response = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        })

        if (response.ok) {
          const result = await response.json()
          console.log("File uploaded:", result)
          uploadedFileName = result.fileName
        } else {
          console.error("File upload failed")
        }
      } catch (error) {
        console.error("Error uploading file:", error)
      }
    }
    
    // 使用自定义提交函数，将文件信息传递给聊天API
    const customSubmit = (e: React.FormEvent<HTMLFormElement>) => {
      handleSubmit(e, {
        options: {
          body: {
            file: uploadedFileName ? { name: uploadedFileName } : null
          }
        }
      })
    }
    
    customSubmit(e)
  }

  return (
    <div className="flex-1 flex flex-col p-4">
      <h1 className="text-2xl font-bold mb-4">{translations.title}</h1>
      <Card className="flex-grow mb-4">
        <CardContent>
          <ScrollArea className="h-[calc(100vh-300px)]">
            {messages.map((message) => (
              <div key={message.id} className={`mb-4 ${message.role === "user" ? "text-right" : "text-left"}`}>
                <span
                  className={`inline-block p-2 rounded-lg ${
                    message.role === "user" ? "bg-blue-500 text-white" : "bg-gray-200 text-black"
                  }`}
                >
                  {message.content}
                </span>
              </div>
            ))}
          </ScrollArea>
        </CardContent>
      </Card>
      <form onSubmit={handleFormSubmit} className="flex gap-2">
        <Input
          type="file"
          onChange={handleFileChange}
          className="flex-grow"
          placeholder={translations.main?.fileUpload}
        />
        <Input
          value={input}
          onChange={handleInputChange}
          placeholder={translations.main?.askQuestion}
          className="flex-grow"
        />
        <Button type="submit">{translations.main?.send}</Button>
      </form>
    </div>
  )
}

