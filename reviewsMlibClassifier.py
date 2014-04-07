from pyspark.mllib.classification import LogisticRegressionWithSGD
from numpy import array
from pyspark import SparkContext


def main():
    # Load and parse the data
    sc = SparkContext("local", "SparkSampleRun")
    data = sc.textFile("sample_reviews.txt")
    parsedData = data.map(lambda line: [x for x in line.split(' ') if x])
    model = LogisticRegressionWithSGD.train(parsedData)

    # Build the model
    labelsAndPreds = parsedData.map(lambda point: (point.item(0),model.predict(point.take(range(1, point.size)))))

    # Evaluating the model on training data
    trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
    print("Training Error = " + str(trainErr))

if __name__ == "__main__":
    main()
