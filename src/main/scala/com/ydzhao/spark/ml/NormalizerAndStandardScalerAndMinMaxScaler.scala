package com.ydzhao.spark.ml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{Normalizer, StandardScaler, MinMaxScaler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object NormalizerAndStandardScalerAndMinMaxScaler {

  Logger.getLogger("org").setLevel(Level.ERROR)

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("NormalizerAndStandardScalerAndMinMaxScaler")
      .getOrCreate()

    // (1) Normalizer
    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.5, -1.0)),
      (1, Vectors.dense(2.0, 1.0, 1.0)),
      (2, Vectors.dense(4.0, 10.0, 2.0))
    )).toDF("id", "features")
    dataFrame.show()

    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)

    val l1NormData = normalizer.transform(dataFrame)
    println("Normalized using L^1 norm")
    l1NormData.show()

    val lInfNormData = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity)
    println("Normalized using L^inf norm")
    lInfNormData.show()

    // (2) StandardScaler
    val dataFrame2 = spark
      .read
      .format("libsvm")
      .load("C:/Users/zhaoyadong/project/sparkml/src/main/resources/sample_libsvm_data.txt")

    val standardscaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)
    val standardscalerModel = standardscaler.fit(dataFrame2)
    val standardscalerData = standardscalerModel.transform(dataFrame2)
    standardscalerData.show()

    // (3) MinMaxScaler
    val dataFrame3 = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -1.0)),
      (1, Vectors.dense(2.0, 1.1, 1.0)),
      (2, Vectors.dense(3.0, 10.1, 3.0))
    )).toDF("id", "features")

    val minmaxscaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    val minmaxscalerModel = minmaxscaler.fit(dataFrame)

    val minmaxscalerData = minmaxscalerModel.transform(dataFrame)
    println(s"Features scaled to range: [${minmaxscaler.getMin}, ${minmaxscaler.getMax}]")
    minmaxscalerData.select("features", "scaledFeatures").show()

    spark.stop()
  }
}
