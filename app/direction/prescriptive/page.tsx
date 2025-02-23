import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function PrescriptiveAnalysisPage() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Prescriptive Analysis</h1>
      <Card>
        <CardHeader>
          <CardTitle>What is Prescriptive Analysis?</CardTitle>
        </CardHeader>
        <CardContent>
          <p>
            Prescriptive analysis goes beyond predicting future outcomes by suggesting actions to benefit from the
            predictions and showing the implications of each decision option. It uses a combination of techniques and
            tools to help businesses decide what to do.
          </p>
          <h3 className="font-semibold mt-4 mb-2">Common techniques include:</h3>
          <ul className="list-disc list-inside">
            <li>Optimization algorithms</li>
            <li>Simulation modeling</li>
            <li>Decision trees</li>
            <li>Game theory</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}

