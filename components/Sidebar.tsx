"use client"

import { useState } from "react"
import { useLanguage } from "@/contexts/LanguageContext"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Search, BarChart, TrendingUp, Zap, HelpCircle } from "lucide-react"

const categories = [
  { name: "all", icon: Search },
  { name: "descriptive", icon: BarChart },
  { name: "predictive", icon: TrendingUp },
  { name: "prescriptive", icon: Zap },
]

const algorithms = [
  { name: "Linear Regression", category: "Predictive" },
  { name: "Logistic Regression", category: "Predictive" },
  { name: "Decision Trees", category: "Predictive" },
  { name: "Random Forests", category: "Predictive" },
  { name: "K-Means Clustering", category: "Descriptive" },
  { name: "Hierarchical Clustering", category: "Descriptive" },
  { name: "Principal Component Analysis", category: "Descriptive" },
  { name: "Support Vector Machines", category: "Predictive" },
  { name: "Neural Networks", category: "Predictive" },
  { name: "Time Series Analysis", category: "Predictive" },
  { name: "Association Rules", category: "Descriptive" },
  { name: "Naive Bayes", category: "Predictive" },
  { name: "K-Nearest Neighbors", category: "Predictive" },
  { name: "Linear Programming", category: "Prescriptive" },
  { name: "Integer Programming", category: "Prescriptive" },
  { name: "Nonlinear Programming", category: "Prescriptive" },
]

export default function Sidebar() {
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedCategory, setSelectedCategory] = useState("all")
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(null)
  const { translations } = useLanguage()

  const filteredAlgorithms = algorithms.filter(
    (algo) =>
      algo.name.toLowerCase().includes(searchTerm.toLowerCase()) &&
      (selectedCategory === "all" || algo.category.toLowerCase() === selectedCategory),
  )

  return (
    <div className="w-64 bg-gray-100 p-4 flex flex-col h-full">
      <h2 className="text-2xl font-bold mb-4">{translations.title}</h2>

      <Input
        type="text"
        placeholder={translations.sidebar?.search}
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className="mb-4"
      />

      <div className="mb-4">
        <h3 className="font-semibold mb-2">{translations.sidebar?.categories}</h3>
        {categories.map((category) => (
          <Button
            key={category.name}
            variant="ghost"
            className="w-full justify-start mb-1"
            onClick={() => setSelectedCategory(category.name)}
          >
            <category.icon className="mr-2 h-4 w-4" />
            {translations.categories?.[category.name]}
          </Button>
        ))}
      </div>

      <div className="flex-grow overflow-auto">
        <h3 className="font-semibold mb-2">{translations.sidebar?.algorithms}</h3>
        {filteredAlgorithms.map((algo) => (
          <Button
            key={algo.name}
            variant="ghost"
            className="w-full justify-start mb-1 text-sm"
            onClick={() => setSelectedAlgorithm(algo)}
          >
            {algo.name}
          </Button>
        ))}
      </div>

      <Button variant="ghost" className="mt-auto justify-start">
        <HelpCircle className="mr-2 h-4 w-4" />
        {translations.sidebar?.help}
      </Button>
    </div>
  )
}

