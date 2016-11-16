package se.kth.spark.lab1.task1

import se.kth.spark.lab1._
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

import scala.collection.mutable.ListBuffer

object Main {

  case class Song(year: Double, f1: Double, f2: Double, f3: Double)

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import org.apache.spark.sql.functions._
    import sqlContext._


    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sc.textFile(filePath).toDF()

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?

    //rdd.take(5).foreach(x=> println(x))
    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(row => row.split(","))

    // recordsRdd.take(5).foreach(x => println(x.mkString(" ")))

    //Step3: map each row into a Song object by using the year label and the first three features
    val songsRdd = recordsRdd.map(row => Song(row(0).toDouble,row(1).toDouble,row(2).toDouble,row(3).toDouble))

    //Step4: convert your rdd into a dataframe
    val songsDf = songsRdd.toDF()//("year","f1","f2","f3")


    // Questions:
    //1. How many are there in the dataset?
    songsDf.registerTempTable("songs")
    //println(songsRdd.count()) // RDD
    println(songsDf.count())  // DataFrame
    sqlContext.sql("SELECT COUNT(*) FROM songs").show()

    //2. How many songs were released between 1998 & 2000

    //println(songsRdd.filter(_.year >= 1998).filter( _.year<=2000).count())
    println(songsDf.filter(songsDf("year") >= 1998 && songsDf("year") <= 2000).count())
    sqlContext.sql("SELECT COUNT(*) FROM songs WHERE year>=1998 and year<=2000") .show()

    //3. min, max and mean of the year column
    val years = songsRdd.map(_.year)
    println(years.min()+" "+years.max()+" "+years.mean()) // RDD
    songsDf.agg(min("year"),max("year"),avg("year")).show() // DataFrame
    sqlContext.sql("SELECT MIN(year),MAX(year),AVG(year) FROM songs").show()

    //4. Show the number of songs per year between the years 2000 and 2010?
    /*
    val sMap =songsRdd.filter(_.year >= 2000).filter( _.year<=2010).map(song => (song.year,1))
    val sReduce = sMap.reduceByKey(_+_)
    sReduce.sortByKey().collect().foreach(println)
    */

    songsDf.filter(songsDf("year") >= 2000 && songsDf("year") <= 2010).groupBy("year").count().orderBy("year").show()
    sqlContext.sql("SELECT year,count(year) FROM songs WHERE year>=2000 and year<=2010 GROUP BY year ORDER BY year").show()
  }
}