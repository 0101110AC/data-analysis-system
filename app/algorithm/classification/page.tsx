import { Card, CardContent, CardHeader, CardTitle } from "../../../components/ui/card"
import { FC } from 'react'

const ClassificationPage: FC = () => {
  return (
    <div className="p-4" role="main" aria-labelledby="page-title">
      <h1 id="page-title" className="text-2xl font-bold mb-4" role="heading" aria-level={1}>Classification Algorithms</h1>
      <Card>
        <CardHeader>
          <CardTitle>What are Classification Algorithms?</CardTitle>
        </CardHeader> 
        <CardContent>
          <p className="mb-4">
            Classification algorithms are a type of supervised learning algorithm used to categorize input data into
            classes. The algorithm learns from labeled training data to create a model that can classify new, unseen
            data.
          </p>
          <h3 className="font-semibold mt-4 mb-2" role="heading" aria-level={2}>Common classification algorithms:</h3>
          <ul className="list-disc list-inside space-y-1" role="list">
            <li role="listitem">Logistic Regression</li>
            <li role="listitem">Decision Trees</li>
            <li role="listitem">Random Forests</li>
            <li role="listitem">Support Vector Machines (SVM)</li>
            <li role="listitem">K-Nearest Neighbors (KNN)</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}

export default ClassificationPage

