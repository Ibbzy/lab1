package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{DenseVector, Matrices, Vector, Vectors}

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    val a1 = v1.toArray
    val a2 = v2.toArray
    a1.zip(a2).map{e:(Double,Double) => e._1 * e._2}.sum
    //return d
  }

  def dot(v: Vector, s: Double): Vector = {
    val a = v.toArray.map{e:(Double) => e*s }
    Vectors.dense(a)
    //return v2
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    val a1 = v1.toArray
    val a2 = v2.toArray
    val a3 = a1.zip(a2).map{e:(Double,Double) => e._1 + e._2}
    Vectors.dense(a3)
    //return v3
  }

  def fill(size: Int, fillVal: Double): Vector = {
    val a = Array.fill(size)(fillVal)
    Vectors.dense(a)
    //return a
  }
}