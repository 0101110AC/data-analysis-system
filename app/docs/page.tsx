import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function DocumentationPage() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Documentation</h1>
      <Card>
        <CardHeader>
          <CardTitle>Data Analysis Assistant Documentation</CardTitle>
        </CardHeader>
        <CardContent>
          <h2 className="text-xl font-semibold mb-2">Getting Started</h2>
          <p className="mb-4">
            Welcome to the Data Analysis Assistant. This tool helps you analyze your data using various statistical and
            machine learning techniques. Here's how to get started:
          </p>
          <ol className="list-decimal list-inside space-y-2 mb-4">
            <li>Upload your data file on the main page.</li>
            <li>Ask questions about your data in natural language.</li>
            <li>Explore different analysis directions and algorithms using the sidebar.</li>
          </ol>

          <h2 className="text-xl font-semibold mb-2">Features</h2>
          <ul className="list-disc list-inside space-y-2 mb-4">
            <li>Natural language interface for data analysis</li>
            <li>Support for various file formats (CSV, JSON, etc.)</li>
            <li>Multiple analysis directions: Descriptive, Predictive, and Prescriptive</li>
            <li>Various algorithms: Regression, Classification, and Clustering</li>
          </ul>

          <h2 className="text-xl font-semibold mb-2">Need Help?</h2>
          <p>
            If you need further assistance, please visit our{" "}
            <a href="/help" className="text-blue-600 hover:underline">
              Help Center
            </a>{" "}
            or contact our support team.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}

