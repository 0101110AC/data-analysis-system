import type React from "react"
import Sidebar from "@/components/Sidebar"
import LanguageSwitcher from "@/components/LanguageSwitcher"
import { useLanguage } from "@/contexts/LanguageContext"

export default function LayoutContent({ children }: { children: React.ReactNode }) {
  const { translations } = useLanguage()

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <header className="h-16 border-b border-gray-200 px-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold">{translations.mainTitle || "Data Analysis Assistant"}</h1>
          <LanguageSwitcher />
        </header>
        <main className="flex-1 p-4">{children}</main>
      </div>
    </div>
  )
}

