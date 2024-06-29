package example

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.SparkSession

class SequenceIndexer(override val uid: String) extends UnaryTransformer[Seq[String], Seq[Int], SequenceIndexer] 
  with DefaultParamsWritable {
  
  /** Dictionary created from [[vocabulary]] and its indices, broadcast once for [[transform()]] */
  private var broadcastDict: Option[Broadcast[Map[String, Int]]] = None
  private var maxLength = 20

  def this(dictionary: Map[String, Int], maxLength: Int = 20) = {
    this(Identifiable.randomUID("seqIdx"))
    if (broadcastDict.isEmpty) {
      broadcastDict = Some(SparkSession.active.sparkContext.broadcast(dictionary))
    }
    this.maxLength = maxLength
  }

  private def f(x: Seq[String]): Seq[Int] = {
    val brDict = broadcastDict.get.value
    val t = x.map(token => brDict.getOrElse(token, 0) + 1)
    if (t.length >= maxLength) 
      t.take(maxLength)
    else {
      // pad with size of dictionary
      t ++ Array.fill(maxLength - t.length)(0)
    }
  }

  override protected def createTransformFunc: Seq[String] => Seq[Int] = f(_)

  override protected def outputDataType: DataType = new ArrayType(IntegerType, true)

  
}
