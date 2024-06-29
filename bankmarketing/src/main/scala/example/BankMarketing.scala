package example

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature.StandardScaler

object BankMarketing {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("BankMarketing").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val df = spark.read.options(Map("delimiter" -> ";", "header" -> "true", "inferSchema" -> "true")).csv("dat/full.csv")
    val integerCols = df.schema.fields.filter(f => f.dataType == IntegerType || f.dataType == DoubleType).map(_.name)


    val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("mIdx")
    val maritalEncoder = new OneHotEncoder().setInputCol("mIdx").setOutputCol("mVector").setDropLast(false)
    val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("eIdx")
    val educationEncoder = new OneHotEncoder().setInputCol("eIdx").setOutputCol("eVector").setDropLast(false)
    val assembler = new VectorAssembler().setInputCols(Array("mVector", "eVector") ++ integerCols).setOutputCol("features")
    val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)  
    val regression = new LogisticRegression().setFeaturesCol("features").setLabelCol("label").setFeaturesCol("scaledFeatures")

    val pipeline = new Pipeline().setStages(Array(maritalIndexer, maritalEncoder, educationIndexer, 
      educationEncoder, assembler, labelIndexer, scaler, regression))

    val Array(training, validation) = df.randomSplit(Array(0.8, 0.2))

    val pipelineModel = pipeline.fit(training)
    // val pipelineModel = PipelineModel.load("bin/model")

    val ef = pipelineModel.transform(training)
    ef.select("marital", "education", "features", "label", "prediction").show(false)
    pipelineModel.write.overwrite.save("bin/model")
    ef.select("rawPrediction", "prediction", "label").show()


    val ff = pipelineModel.transform(validation)

    val evaluator = new BinaryClassificationEvaluator()
    val areaUnderROC = evaluator.evaluate(ef)
    val areaUnderROCValidation = evaluator.evaluate(ff)

    println(s"areaUnderROC (training) = $areaUnderROC")
    println(s"areaUnderROC (validation) = $areaUnderROCValidation")

    spark.stop()
  }
}

// cis503staff