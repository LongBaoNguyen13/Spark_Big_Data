package example

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineStage}

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.types.{IntegerType, DoubleType}
import breeze.plot._


object newBankMartketing {
    val spark = SparkSession.builder
        .appName("Pipeline Example")
        .master("local[*]")
        .getOrCreate()

    // Bai 1

    // cau a
    val df = spark.read.options(Map("delimiter" -> ";", "header" -> "true", "inferSchema" -> "true")).csv("data/bank_marketing.csv")
    df.printSchema()
    val head = df.take(20)
    head.foreach(println)

    // cau b
    df.groupBy("education").count().orderBy($"count".desc).show()

    // cau c
    df.groupBy("age").count().orderBy($"count".desc).show()

    // cau d
    val summary = df.describe()
    summary.show()
    summary.select("summary", "age", "job", "marital", "education").show()

    // cau e
    // import org.apache.spark.sql.types._
    val integerCols = df.schema.fields.filter(f => f.dataType == IntegerType || f.dataType == DoubleType).map(_.name)
    val stringCols = df.schema.fields.filter(f => f.dataType == StringType).map(_.name)

    // Bai 2
    // cau a
    // import org.apache.spark.ml.feature.StringIndexer
    val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("mIdx")
    val model = maritalIndexer.fit(df)
    val df1 = model.transform(df)
    df1.show()

    // cau b
    // import org.apache.spark.ml.feature.OneHotEncoder
    val maritalEncoder = new OneHotEncoder().setInputCol("mIdx").setOutputCol("mVector")
    val df2 = maritalEncoder.fit(df1).transform(df1)
    df2.show()
    df2.printSchema()
    df2.select("mVector").show(false)

    // cau c
    // Transform education column
    val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("eIdx")
    val df3 = educationIndexer.fit(df2).transform(df2)
    df3.show()

    val educationEncoder = new OneHotEncoder().setInputCol("eIdx").setOutputCol("eVector")
    val df4 = educationEncoder.fit(df3).transform(df3)
    df4.show()

    // cau d
    // import org.apache.spark.ml.feature.VectorAssembler
    val assembler = new VectorAssembler().setInputCols(Array("mVector", "eVector") ++ integerCols).setOutputCol("features'")
    val df5 = assembler.transform(df4)
    df5.printSchema()
    df5.show()

    // cau e
    // Create the pipeline with the above stages
    val pipeline = new Pipeline()
    .setStages(Array[PipelineStage](maritalIndexer, maritalEncoder, educationIndexer, educationEncoder, assembler))

    // Fit the pipeline to the data
    val df5 = pipeline.fit(df).transform(df)

    // Show the resulting DataFrame df5
    df5.show(false)

    // Inspect the schema to ensure the features column has been created correctly
    df5.printSchema()

    // cau f
    val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
    val pipeline = new Pipeline().setStages(Array[PipelineStage](labelIndexer))

    val df6 = pipeline.fit(df5).transform(df5)

    df6.show(false)
    df6.printSchema()

    // cau g
    // Remove the existing 'features' column if it exists in df6
    val df6 = df5.drop("features")

    // Get the columns to be assembled
    val newIntegerCols = df6.schema.fields.filter(f => f.dataType == IntegerType || f.dataType == DoubleType).map(_.name)

    // Create a new VectorAssembler
    val newAssembler = new VectorAssembler().setInputCols(newIntegerCols).setOutputCol("newFeatures")

    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)

    // cau h
    // Step 7: LogisticRegression model
    val lr = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setFeaturesCol("scaledFeatures")
      .setMaxIter(10)
      .setRegParam(0.01)

    // Create the pipeline with the above stages
    val pipeline = new Pipeline().setStages(Array[PipelineStage](maritalIndexer, maritalEncoder, educationIndexer, educationEncoder, assembler, labelIndexer, scaler, lr))

    val Array(training, validation) = df.randomSplit(Array(0.8, 0.2))

    val pipelineModel = pipeline.fit(training)
    val ef = pipelineModel.transform(training)
    ef.select("marital", "education", "features", "label", "prediction").show(false)
    pipelineModel.write.overwrite.save("bin/model")
    // import org.apache.spark.ml.PipelineMode
    val pipelineModel = PipelineModel.load("bin/model")
    ef.select("rawPrediction", "prediction", "label").show()

    val ff = pipelineModel.transform(validation)
    val evaluator = new BinaryClassificationEvaluator()

    val areaUnderROC = evaluator.evaluate(ef)
    val areaUnderROCValidation = evaluator.evaluate(ff)
    

    // Compute F-measure for different thresholds
    val logisticRegressionModel = pipelineModel.stages.last.asInstanceOf[LogisticRegressionModel]
    val thresholds = (0.0 to 1.0 by 0.05).toArray
    val metrics = thresholds.map { t =>
    val predictions = logisticRegressionModel.setThreshold(t).transform(ff)
    val tp = predictions.filter($"prediction" === 1.0 && $"labelIdx" === 1.0).count.toDouble
    val fp = predictions.filter($"prediction" === 1.0 && $"labelIdx" === 0.0).count.toDouble
    val fn = predictions.filter($"prediction" === 0.0 && $"labelIdx" === 1.0).count.toDouble
    val precision = if (tp + fp == 0) 0 else tp / (tp + fp)
    val recall = if (tp + fn == 0) 0 else tp / (tp + fn)
    val fMeasure = if (precision + recall == 0) 0 else 2 * precision * recall / (precision + recall)
    (t, precision, recall, fMeasure)
    }

    val bestThreshold = metrics.maxBy(_._4)._1
    val bestFMeasure = metrics.maxBy(_._4)._4

    println(s"Best F-measure: $bestFMeasure at threshold $bestThreshold")


}





import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.types.{IntegerType, DoubleType}
import breeze.plot._

val spark = SparkSession.builder
    .appName("Pipeline with LogisticRegression")
    .master("local[*]")
    .getOrCreate()

import spark.implicits._


// Step 1: StringIndexer for the 'marital' column
val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("mIdx")

// Step 2: OneHotEncoder for the 'mIdx' column
val maritalEncoder = new OneHotEncoder().setInputCol("mIdx").setOutputCol("mVector")

// Step 3: StringIndexer for the 'education' column
val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("eIdx")

// Step 4: OneHotEncoder for the 'eIdx' column
val educationEncoder = new OneHotEncoder().setInputCol("eIdx").setOutputCol("eVector")

// Step 5: VectorAssembler to combine 'mVector', 'eVector', and integer columns
val integerCols = df.schema.fields.filter(f => f.dataType == IntegerType || f.dataType == DoubleType).map(_.name)
val assembler = new VectorAssembler().setInputCols(Array("mVector", "eVector") ++ integerCols).setOutputCol("features")

// Step 6: MinMaxScaler to scale the features
val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")

// Step 7: StringIndexer for the 'label' column to convert it to numeric format
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("labelIdx")

// Step 8: LogisticRegression model
val lr = new LogisticRegression()
  .setLabelCol("labelIdx")
  .setFeaturesCol("scaledFeatures")
  .setMaxIter(10)
  .setRegParam(0.01)

// Create the pipeline with the above stages
val pipeline = new Pipeline().setStages(Array[PipelineStage](maritalIndexer, maritalEncoder, educationIndexer, educationEncoder, assembler, scaler, labelIndexer, lr))

// Split the data into training and validation sets
val Array(training, validation) = df.randomSplit(Array(0.8, 0.2))

// Fit the pipeline to the training data
val pipelineModel = pipeline.fit(training)

// Transform the training data using the fitted pipeline
val trainTransformed = pipelineModel.transform(training)
trainTransformed.select("marital", "education", "features", "scaledFeatures", "label", "labelIdx", "prediction").show(false)

// Save the pipeline model
pipelineModel.write.overwrite.save("bin/model")

// Load the saved pipeline model
val loadedPipelineModel = PipelineModel.load("bin/model")

// Transform the validation data using the loaded pipeline model
val validationTransformed = loadedPipelineModel.transform(validation)

// Evaluate the model
val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("labelIdx")
  .setRawPredictionCol("rawPrediction")

val areaUnderROC = evaluator.evaluate(trainTransformed)
val areaUnderROCValidation = evaluator.evaluate(validationTransformed)

println(s"Area Under ROC for training data = $areaUnderROC")
println(s"Area Under ROC for validation data = $areaUnderROCValidation")

// Compute F-measure for different thresholds
val logisticRegressionModel = pipelineModel.stages.last.asInstanceOf[LogisticRegressionModel]
val thresholds = (0.0 to 1.0 by 0.05).toArray
val metrics = thresholds.map { t =>
  val predictions = logisticRegressionModel.setThreshold(t).transform(validationTransformed.drop("prediction"))
  val tp = predictions.filter($"prediction" === 1.0 && $"labelIdx" === 1.0).count.toDouble
  val fp = predictions.filter($"prediction" === 1.0 && $"labelIdx" === 0.0).count.toDouble
  val fn = predictions.filter($"prediction" === 0.0 && $"labelIdx" === 1.0).count.toDouble
  val precision = if (tp + fp == 0) 0 else tp / (tp + fp)
  val recall = if (tp + fn == 0) 0 else tp / (tp + fn)
  val fMeasure = if (precision + recall == 0) 0 else 2 * precision * recall / (precision + recall)
  (t, precision, recall, fMeasure)
}

val bestThreshold = metrics.maxBy(_._4)._1
val bestFMeasure = metrics.maxBy(_._4)._4

println(s"Best F-measure: $bestFMeasure at threshold $bestThreshold")

// Plot F-measures vs. Thresholds
val fig = Figure()
val plt = fig.subplot(0)
plt += plot(thresholds, metrics.map(_._4))
plt.xlabel = "Threshold"
plt.ylabel = "F-measure"
plt.title = "F-measure vs. Threshold"
fig.refresh()

// Perform 5-Fold Cross-Validation
val paramGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(0.01, 0.1, 0.5))
  .addGrid(lr.maxIter, Array(10, 20))
  .build()

val crossValidator = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator().setLabelCol("labelIdx"))
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)

val cvModel = crossValidator.fit(training)
val cvTransformed = cvModel.transform(validation)
val auc = evaluator.evaluate(cvTransformed)

println(s"Area Under ROC after cross-validation = $auc")

