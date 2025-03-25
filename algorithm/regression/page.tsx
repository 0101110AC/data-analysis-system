import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function RegressionPage() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Regression Analysis</h1>
      <Card>
        <CardHeader>
          <CardTitle>What is Regression Analysis?</CardTitle>
        </CardHeader>
        <CardContent>
          <p>
            Regression analysis is a set of statistical methods used for estimating the relationships between a
            dependent variable and one or more independent variables. It can be used to make predictions and to
            understand which variables have the biggest effect on the dependent variable.
          </p>
          <h3 className="font-semibold mt-4 mb-2">Common types of regression:</h3>
          <ul className="list-disc list-inside">
            <li>Linear Regression</li>
            <li>Multiple Regression</li>
            <li>Polynomial Regression</li>
            <li>Logistic Regression</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}

