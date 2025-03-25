import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function ClusteringPage() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Clustering Algorithms</h1>
      <Card>
        <CardHeader>
          <CardTitle>What are Clustering Algorithms?</CardTitle>
        </CardHeader>
        <CardContent>
          <p>
            Clustering algorithms are unsupervised learning methods that group similar data points together based on
            their features. The goal is to find inherent structures in the data without using pre-existing labels.
          </p>
          <h3 className="font-semibold mt-4 mb-2">Common clustering algorithms:</h3>
          <ul className="list-disc list-inside">
            <li>K-Means</li>
            <li>Hierarchical Clustering</li>
            <li>DBSCAN (Density-Based Spatial Clustering of Applications with Noise)</li>
            <li>Gaussian Mixture Models</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}

