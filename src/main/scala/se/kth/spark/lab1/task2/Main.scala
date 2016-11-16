package se.kth.spark.lab1.task2

import se.kth.spark.lab1._
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SQLContext


object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sc.textFile(filePath).toDF("raw")
    //rawDF.show(3)

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("raw")
      .setOutputCol("words")
      .setPattern("\\W")

    //Step2: transform with tokenizer and show 5 rows
    val regDF  = regexTokenizer.transform(rawDF)
    //regDF.limit(5).show()
    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)

    val arr2Vect = new Array2Vector()
      .setInputCol("words")
      .setOutputCol("vector")


    val vecDf = arr2Vect.transform(regDF)
    //vecDf.show(3)


    //vRegDf.rdd.map(row =)
    //vRegDf.select(vRegDf.col("vector").
    //vRegDf.createOrReplaceTempView("table")
    //sqlContext.udf.register("yearUDF", (x:Vector[String]) => x.apply(0))
    //val query = "SELECT yearUDF(vector) AS year FROM table"
    //sql(query).show()
    // vRegDf.withColumn("firstValue", callUDF("yearUDF",vRegDf.col("vector")))
    //vRegDf.select($"id", callUDF("yearUDF", $"value"))

    //Step4: extract the label(year) into a new column
    //vRegDf.map({ case Row(v: Vector[String]) => v(0) })
    //val r = vRegDf.rdd

    //val year = r.map(row => row.getAs[DenseVector]("vector").values(0)).toDF("year")
    //val yearvec= vec("year")
    //println(year.dtypes(0))
    //print(vec.count())
    //print(vRegDf.count())
    //val lSlicer = vec
    //vRegDf.withColumn("year",yearvec)
    //vRegDf.drop("raw","words").collect()
    val lSlicer = new VectorSlicer()
      .setInputCol("vector")
      .setOutputCol("labelV")
      .setIndices(Array(0))

    val lDf = lSlicer.transform(vecDf)

    //lDf.show(3)

    //Step5: convert type of the label from vector to double (use our Vector2Double)
    //val yo=udf {(s:DenseVector) => s.values(0)}
    val v2d = new Vector2DoubleUDF({(v: Vector) => v.apply(0)})
      .setInputCol("labelV")
      .setOutputCol("labelD")

    val dDf = v2d.transform(lDf)
    //dDf.show(3)



    //
    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)

    val minYear = 1922.0
    val lShifter = new DoubleUDF({(d: Double) => d-minYear})
      .setInputCol("labelD")
      .setOutputCol("label")

    val sDf = lShifter.transform(dDf)
    //sDf.show(3)

    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer()
      .setInputCol("vector")
      .setOutputCol("features")
      .setIndices(Array(1,2,3))

    val fDf = fSlicer.transform(sDf)



    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer))

    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)

    //Step10: transform data with the model - do predictions
    val pDF = pipelineModel.transform(rawDF)

    //Step11: drop all columns from the dataframe other than label and features
    val finalDF = pDF.select("features","label")
    finalDF.show(3)
  }
}