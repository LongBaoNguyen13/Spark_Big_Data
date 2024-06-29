package example

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.NGram
import org.apache.spark.ml.feature.CountVectorizerModel

import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.layers.Embedding
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.layers.LSTM
import com.intel.analytics.bigdl.dllib.keras.layers.Dense
import com.intel.analytics.bigdl.dllib.keras.layers.SoftMax
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.keras.layers.Dropout
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.dllib.keras.metrics.Accuracy
import com.intel.analytics.bigdl.dllib.visualization.TrainSummary
import com.intel.analytics.bigdl.dllib.visualization.ValidationSummary
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.optim.Trigger
import org.apache.spark.SparkConf
import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.optim.Loss
import com.intel.analytics.bigdl.dllib.optim.Top1Accuracy


object Tweeter {


  private def createModel(vocabSize: Int, maxLength: Int): Sequential[Float] = {
    val model = Sequential()
    model.add(Embedding(vocabSize + 1, 16, inputLength = maxLength))
    model.add(LSTM(32))
    model.add(Dropout(0.1))
    model.add(Dense(3))
    model.add(SoftMax())
    model
  }

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName(getClass.getName).setMaster("local[*]")
    val sc = NNContext.initNNContext(conf)
    sc.setLogLevel("ERROR")
    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()

    val df = spark.read.parquet("Lab03/sa.parquet")
    val Array(uf, vf) = df.randomSplit(Array(0.8, 0.2), seed = 1234)

    df.show()

    val tokenizer = new Tokenizer().setInputCol("tweet").setOutputCol("tokens")

    // val ngrams = new NGram().setInputCol("tokens").setOutputCol("bigrams").setN(2)
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("x").setMinDF(5)
    // val idf = new IDF().setInputCol("x").setOutputCol("features")
    // val logreg = new LogisticRegression().setFeaturesCol("features").setLabelCol("sentiment")
    // val pipeline = new Pipeline().setStages(Array(tokenizer, vectorizer, idf, logreg))

    val preprocessor = new Pipeline().setStages(Array(tokenizer, vectorizer))
    val model = preprocessor.fit(uf) // train on 80% of the dataset

    val ef = model.transform(uf) 
    // create a dictionary ["token" -> index]
    val vocab = model.stages(1).asInstanceOf[CountVectorizerModel].vocabulary
    val dictionary = vocab.zipWithIndex.toMap

    val maxLength = 30
    val seqIndexer = new SequenceIndexer(dictionary, maxLength).setInputCol("tokens").setOutputCol("features")
    val ff = seqIndexer.transform(ef)

    import org.apache.spark.sql.functions._

    val gf = ff.withColumn("label", col("sentiment") + 1f)
    gf.select("features", "sentiment", "label").show(false)

    val bigdl = createModel(vocab.size, maxLength)
    bigdl.summary()
    val estimator = NNEstimator(bigdl, ClassNLLCriterion(logProbAsInput = false), featureSize = Array(maxLength), labelSize = Array(1))
    val trainingSummary = TrainSummary(appName = "twitter", logDir = "sum/")
    val validationSummary = ValidationSummary(appName = "twitter", logDir = "sum/")
    
    estimator.setBatchSize(128)
        .setOptimMethod(new Adam(lr = 5E-5))
        .setMaxEpoch(30)
        .setTrainSummary(trainingSummary)
        .setValidationSummary(validationSummary)
        .setValidation(Trigger.everyEpoch, gf, Array(new Top1Accuracy(), new Loss()), 128)

    val model2 = estimator.fit(gf)
    model2.write.overwrite().save("bin/lstm.bigdl")

    // ef.select("sentiment", "prediction").show()

    // val evaluator = new MulticlassClassificationEvaluator().setLabelCol("sentiment")
    // val f1u = evaluator.evaluate(ef)
    // println(s"F1 score (training) = $f1u")

    // val f1v = evaluator.evaluate(model.transform(vf))
    // println(s"F1 score (test) = $f1v")

    spark.stop()
  }
}
