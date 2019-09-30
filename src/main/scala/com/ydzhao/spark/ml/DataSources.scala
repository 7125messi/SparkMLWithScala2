package com.ydzhao.spark.ml

import org.apache.log4j.{Level,Logger}
import java.util.Properties
import org.apache.spark.sql.SparkSession

/**
 * Beside some general data sources such as Parquet, CSV, JSON and JDBC
 * https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/sql/SQLDataSourceExample.scala
 */
object DataSources {

  Logger.getLogger("org").setLevel(Level.ERROR)

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Spark SQL data sources example")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

//    runBasicDataSourceExample(spark)
//    runBasicParquetExample(spark)
//    runParquetSchemaMergingExample(spark)
//    runJsonDatasetExample(spark)
    runJdbcDatasetExample(spark)

    spark.stop()
  }

  private def runBasicDataSourceExample(spark: SparkSession): Unit = {
    val usersDF = spark.read.load("C:/Users/zhaoyadong/project/sparkml/src/main/resources/users.parquet")
    usersDF.select("name", "favorite_color").write.save("C:/Users/zhaoyadong/project/sparkml/src/main/resources/namesAndFavColors.parquet")

    val peopleDF = spark.read.format("json").load("C:/Users/zhaoyadong/project/sparkml/src/main/resources/people.json")
    peopleDF.select("name", "age").write.format("parquet").save("C:/Users/zhaoyadong/project/sparkml/src/main/resources/namesAndAges.parquet")

    val peopleDFCsv = spark.read.format("csv")
      .option("sep", ";")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("C:/Users/zhaoyadong/project/sparkml/src/main/resources/people.csv")
//    val sqlDF = spark.sql("SELECT * FROM parquet.`examples/src/main/resources/users.parquet`")
//    peopleDF.write.bucketBy(42, "name").sortBy("age").saveAsTable("people_bucketed")
//    usersDF.write.partitionBy("favorite_color").format("parquet").save("namesPartByColor.parquet")
//    usersDF
//      .write
//      .partitionBy("favorite_color")
//      .bucketBy(42, "name")
//      .saveAsTable("users_partitioned_bucketed")
//
//    spark.sql("DROP TABLE IF EXISTS people_bucketed")
//    spark.sql("DROP TABLE IF EXISTS users_partitioned_bucketed")
  }

  private def runBasicParquetExample(spark: SparkSession): Unit = {
    import spark.implicits._

    val peopleDF = spark.read.json("examples/src/main/resources/people.json")
    peopleDF.write.parquet("people.parquet")

    val parquetFileDF = spark.read.parquet("people.parquet")
    parquetFileDF.createOrReplaceTempView("parquetFile")

    val namesDF = spark.sql("SELECT name FROM parquetFile WHERE age BETWEEN 13 AND 19")
    namesDF.map(attributes => "Name: " + attributes(0)).show()
  }

  private def runParquetSchemaMergingExample(spark: SparkSession): Unit = {
    import spark.implicits._

    val squaresDF = spark.sparkContext.makeRDD(1 to 5).map(i => (i, i * i)).toDF("value", "square")
    squaresDF.write.parquet("data/test_table/key=1")

    val cubesDF = spark.sparkContext.makeRDD(6 to 10).map(i => (i, i * i * i)).toDF("value", "cube")
    cubesDF.write.parquet("data/test_table/key=2")

    val mergedDF = spark.read.option("mergeSchema", "true").parquet("data/test_table")
    mergedDF.printSchema()
  }

  private def runJsonDatasetExample(spark: SparkSession): Unit = {
    import spark.implicits._

    val path = "examples/src/main/resources/people.json"
    val peopleDF = spark.read.json(path)

    peopleDF.printSchema()

    peopleDF.createOrReplaceTempView("people")

    val teenagerNamesDF = spark.sql("SELECT name FROM people WHERE age BETWEEN 13 AND 19")
    teenagerNamesDF.show()

    val otherPeopleDataset = spark.createDataset(
      """{"name":"Yin","address":{"city":"Columbus","state":"Ohio"}}""" :: Nil)
    val otherPeople = spark.read.json(otherPeopleDataset)
    otherPeople.show()
  }

  private def runJdbcDatasetExample(spark: SparkSession): Unit = {
    val jdbcDF = spark.read
      .format("jdbc")
      .option("url", "jdbc:mysql://10.64.56.92:3306/xiaohongshu_hack")
      .option("dbtable", "xhs_hot_keywords")
      .option("driver","com.mysql.jdbc.Driver")
      .option("user", "shihuomain")
      .option("password", "E67uD100")
      .load()
    jdbcDF.show(4)

    // 写数据到表
//    jdbcDF.write
//      .format("jdbc")
//      .option("url", "jdbc:mysql://10.64.56.92:3306/xiaohongshu_hack")
//      .option("dbtable", "xhs_hot_keywords")
//      .option("driver","com.mysql.jdbc.Driver")
//      .option("user", "shihuomain")
//      .option("password", "E67uD100")
//      .save()
  }
}
