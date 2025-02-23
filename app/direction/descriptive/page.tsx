import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function DescriptiveAnalysisPage() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Descriptive Analysis</h1>
      <Card>
        <CardHeader>
          <CardTitle>What is Descriptive Analysis?</CardTitle>
        </CardHeader>
        <CardContent>
          <p>
            Descriptive analysis is the process of using statistical techniques to describe or summarize a set of data.
            This type of analysis helps to understand the basic features of the data and provides simple summaries about
            the sample and the measures.
          </p>
          <h3 className="font-semibold mt-4 mb-2">Common techniques include:</h3>
          <ul className="list-disc list-inside">
            <li>Mean, median, mode</li>
            <li>Standard deviation</li>
            <li>Frequency distributions</li>
            <li>Data visualization (histograms, bar charts, etc.)</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}

