package se.kth.spark.lab1.task3

import org.apache.spark._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import org.apache.spark.ml.linalg.Vector
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF: DataFrame = sc.textFile(filePath).toDF()

    // split DataFrame into test and training data
    val Array(rawtrainDF, rawtestDF) = rawDF.randomSplit(Array(0.7, 0.3))

    // tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("words")
      .setPattern(",").setGaps(true)

    // transform array of tokens to a vector of tokens (use our ArrayToVector)

    val arr2Vect = new Array2Vector()
      .setInputCol("words")
      .setOutputCol("vector")

    val lSlicer = new VectorSlicer()
      .setInputCol("vector")
      .setOutputCol("labelV")
      .setIndices(Array(0))

    // convert type of the label from vector to double (use our Vector2Double)

    val v2d = new Vector2DoubleUDF({(v: Vector) => v.apply(0)})
      .setInputCol("labelV")
      .setOutputCol("labelD")

    // shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)

    val minYear = 1922.0
    val lShifter = new DoubleUDF({(d: Double) => d-minYear})
      .setInputCol("labelD")
      .setOutputCol("label")

    // extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer()
      .setInputCol("vector")
      .setOutputCol("features")
      .setIndices(Array(1,2,3))

    // linear regression setup
    val myLR = new LinearRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setRegParam(0.1)
      .setElasticNetParam(0.1)

    // pipeline setup, one for feature extraction, one for linear regression
    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer,myLR))
    val pipelineModel: PipelineModel = pipeline.fit(rawtrainDF)
    val pipeT = new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer))
    val pipeTModel: PipelineModel = pipeT.fit(rawtrainDF)
    val lrModel = pipelineModel.stages.last.asInstanceOf[LinearRegressionModel]

    //print rmse of our model
    //do prediction on test data - print first k
    val testDF = pipeTModel.transform(rawtestDF)
    val trainingSummary = lrModel.summary
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    val predDF = lrModel.transform(testDF)
    predDF.select("features","prediction","label").show(10)




  }
}