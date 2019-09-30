package com.ydzhao.spark.ml

import org.apache.log4j.{Level,Logger}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

object TfIdfFeatureExtractors {
    Logger.getLogger("org").setLevel(Level.ERROR)
    def main(args: Array[String]) {
      val spark = SparkSession
        .builder
        .master("local")
        .appName("TfIdfExample")
        .getOrCreate()

      val sentenceData = spark.createDataFrame(Seq(
        (0.0, "Hi I heard about Spark"),
        (0.0, "I wish Java could use case classes"),
        (1.0, "Logistic regression models are neat")
      )).toDF("label", "sentence")

      val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
      val wordsData = tokenizer.transform(sentenceData)

      val hashingTF = new HashingTF()
        .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

      val featurizedData = hashingTF.transform(wordsData)

      val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
      val idfModel = idf.fit(featurizedData)

      val rescaledData = idfModel.transform(featurizedData)
      rescaledData.select("label", "features").show()

      spark.stop()
    }
}
