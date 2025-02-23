"use client"

import type React from "react"
import { createContext, useState, useContext, useEffect } from "react"

type LanguageContextType = {
  language: string
  setLanguage: (lang: string) => void
  translations: Record<string, any>
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined)

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [language, setLanguage] = useState("en")
  const [translations, setTranslations] = useState({})

  useEffect(() => {
    const storedLang = localStorage.getItem("language") || "en"
    setLanguage(storedLang)
  }, [])

  useEffect(() => {
    const loadTranslations = async () => {
      const response = await fetch(`/locales/${language}.json`)
      const data = await response.json()
      setTranslations(data)
    }

    loadTranslations()
  }, [language])

  const value = {
    language,
    setLanguage: (lang: string) => {
      localStorage.setItem("language", lang)
      setLanguage(lang)
    },
    translations,
  }

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
}

export const useLanguage = () => {
  const context = useContext(LanguageContext)
  if (context === undefined) {
    throw new Error("useLanguage must be used within a LanguageProvider")
  }
  return context
}

