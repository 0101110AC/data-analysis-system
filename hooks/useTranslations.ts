"use client"

import { useState, useEffect } from "react"

export function useTranslations() {
  const [translations, setTranslations] = useState({})
  const [currentLang, setCurrentLang] = useState("en")

  useEffect(() => {
    const loadTranslations = async () => {
      const lang = localStorage.getItem("language") || "en"
      setCurrentLang(lang)
      const response = await fetch(`/locales/${lang}.json`)
      const data = await response.json()
      setTranslations(data)
    }

    loadTranslations()

    window.addEventListener("languageChange", loadTranslations)
    return () => window.removeEventListener("languageChange", loadTranslations)
  }, [])

  return { translations, currentLang }
}

