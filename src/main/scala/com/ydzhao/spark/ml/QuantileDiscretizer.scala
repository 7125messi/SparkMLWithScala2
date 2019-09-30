package com.ydzhao.spark.ml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.QuantileDiscretizer
import org.apache.spark.sql.SparkSession

object QuantileDiscretizer {
  Logger.getLogger("org").setLevel(Level.ERROR)
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("QuantileDiscretizerExample")
      .getOrCreate()

    val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
    println(data.toBuffer)
    val df = spark
      .createDataFrame(data)
      .toDF("id", "hour")
      .repartition(1)

    // hour is a continuous feature with Double type.
    // We want to turn the continuous feature into a categorical one.
    // Given numBuckets = 3, we should get the following DataFrame
    val discretizer = new QuantileDiscretizer()
      .setInputCol("hour")
      .setOutputCol("result")
      .setNumBuckets(3)

    val result = discretizer.fit(df).transform(df)
    result.show(false)

    spark.stop()
  }
}
