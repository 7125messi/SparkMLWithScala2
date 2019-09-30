package com.ydzhao.spark.ml

/**
 * 该模型为词汇表上的文档生成稀疏表示，然后可以将其传递给其他算法，例如LDA
 * 词频向量化
 * 类似于：from sklearn.feature_extraction.text import CountVectorizer
 */

import org.apache.log4j.{Logger,Level}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.SparkSession

object CountVectorizer {
  Logger.getLogger("org").setLevel(Level.ERROR)
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("CountVectorizer")
      .getOrCreate()

    val df = spark.createDataFrame(Seq(
      (0, Array("a", "b", "c")),
      (1, Array("a", "b", "b", "c", "a"))
    )).toDF("id", "words")

    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(df)

    val cvm = new CountVectorizerModel(Array("a", "b", "c"))
      .setInputCol("words")
      .setOutputCol("features")

    cvModel.transform(df).show(false)
    spark.stop()
  }
}
