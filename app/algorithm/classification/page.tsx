import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function ClassificationPage() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Classification Algorithms</h1>
      <Card>
        <CardHeader>
          <CardTitle>What are Classification Algorithms?</CardTitle>
        </CardHeader>
        <CardContent>
          <p>
            Classification algorithms are a type of supervised learning algorithm used to categorize input data into
            classes. The algorithm learns from labeled training data to create a model that can classify new, unseen
            data.
          </p>
          <h3 className="font-semibold mt-4 mb-2">Common classification algorithms:</h3>
          <ul className="list-disc list-inside">
            <li>Logistic Regression</li>
            <li>Decision Trees</li>
            <li>Random Forests</li>
            <li>Support Vector Machines (SVM)</li>
            <li>K-Nearest Neighbors (KNN)</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}

