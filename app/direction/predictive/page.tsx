import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function PredictiveAnalysisPage() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Predictive Analysis</h1>
      <Card>
        <CardHeader>
          <CardTitle>What is Predictive Analysis?</CardTitle>
        </CardHeader>
        <CardContent>
          <p>
            Predictive analysis uses statistical algorithms and machine learning techniques to identify the likelihood
            of future outcomes based on historical data. It's all about forecasting what might happen in the future.
          </p>
          <h3 className="font-semibold mt-4 mb-2">Common techniques include:</h3>
          <ul className="list-disc list-inside">
            <li>Regression analysis</li>
            <li>Time series forecasting</li>
            <li>Classification algorithms</li>
            <li>Neural networks</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}

