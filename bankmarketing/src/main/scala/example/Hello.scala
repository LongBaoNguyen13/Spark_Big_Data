package example

object BankMarket {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("BankMarketing").master("local[*]").getOrCreate()
    spark.stop()
  }
}
