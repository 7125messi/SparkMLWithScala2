package com.ydzhao.spark.ml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{Matrix, Vectors, Vector}
import org.apache.spark.ml.stat.{Correlation,ChiSquareTest,Summarizer}
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

object BasicStatistics {
  Logger.getLogger("org").setLevel(Level.ERROR)
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("BasicStatistics")
      .getOrCreate()
    import spark.implicits._

    // (1) Correlation相关性
    val data1 = Seq(
      Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
      Vectors.dense(4.0, 5.0, 0.0, 3.0),
      Vectors.dense(6.0, 7.0, 0.0, 8.0),
      Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
    )

    val df1 = data1.map(Tuple1.apply).toDF("features")
    val Row(coeff1: Matrix) = Correlation.corr(df1, "features").head
    println(s"Pearson correlation matrix:\n $coeff1")

    val Row(coeff2: Matrix) = Correlation.corr(df1, "features", "spearman").head
    println(s"Spearman correlation matrix:\n $coeff2")

    // (2) Hypothesis testing假设检验
    val data2 = Seq(
      (0.0, Vectors.dense(0.5, 10.0)),
      (0.0, Vectors.dense(1.5, 20.0)),
      (1.0, Vectors.dense(1.5, 30.0)),
      (0.0, Vectors.dense(3.5, 30.0)),
      (0.0, Vectors.dense(3.5, 40.0)),
      (1.0, Vectors.dense(3.5, 40.0))
    )

    val df2 = data2.toDF("label", "features")
    val chi = ChiSquareTest.test(df2, "features", "label").head
    println(s"pValues = ${chi.getAs[Vector](0)}")
    println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
    println(s"statistics ${chi.getAs[Vector](2)}")

    // (3) Summarizer
    import spark.implicits._
    import Summarizer._

    val data3 = Seq(
      (Vectors.dense(2.0, 3.0, 5.0), 1.0),
      (Vectors.dense(4.0, 6.0, 7.0), 2.0)
    )
    val df3 = data3.toDF("features", "weight")
    val (meanVal, varianceVal) = df3.select(metrics("mean", "variance")
      .summary($"features", $"weight").as("summary"))
      .select("summary.mean", "summary.variance")
      .as[(Vector, Vector)].first()
    println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")
    val (meanVal2, varianceVal2) = df3.select(mean($"features"), variance($"features"))
      .as[(Vector, Vector)].first()
    println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")

    spark.stop()
  }
}
