import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function HelpPage() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Help Center</h1>
      <Card>
        <CardHeader>
          <CardTitle>How to use the Data Analysis Assistant</CardTitle>
        </CardHeader>
        <CardContent>
          <ol className="list-decimal list-inside space-y-2">
            <li>Upload your data file using the file input on the main page.</li>
            <li>Ask questions about your data in the text input field.</li>
            <li>The AI will analyze your data and provide insights based on your questions.</li>
            <li>Explore different analysis directions and algorithms using the sidebar menu.</li>
            <li>Refer to the documentation for detailed information on available features and algorithms.</li>
          </ol>
        </CardContent>
      </Card>
    </div>
  )
}

