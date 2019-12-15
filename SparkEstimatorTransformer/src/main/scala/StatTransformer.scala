import org.apache.spark.ml.Model
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable}
import org.apache.spark.sql
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{Metadata, StructType}
import org.json4s.{Extraction, FullTypeHints}
import org.json4s.jackson.JsonMethods.{compact, parse, render}
import org.json4s.jackson.Serialization

case class ColumnMetadata(mean:Double = 0,
                          max:Double = 0,
                          min:Double = 0,
                          percentile10:Double = 0,
                          percentile50:Double = 0,
                          percentile90:Double = 0,
                          percentile99:Double = 0,
                          uniqCount:Long = 0,
                          std:Double = 0,
                          pirsonCorrelation:Double = 0,
                          spearmanCorrelation:Double = 0){


  def metadata(): Metadata ={
    val metadata:Metadata = new sql.types.MetadataBuilder()
      .putDouble("mean", mean)
      .putDouble("max", max)
      .putDouble("min", min)
      .putDouble("per10", percentile10)
      .putDouble("per50", percentile50)
      .putDouble("per90", percentile90)
      .putDouble("per99", percentile99)
      .putLong("uniqCount", uniqCount)
      .putDouble("std", std)
      .putDouble("pirsonCorrelation", pirsonCorrelation)
      .putDouble("spearmanCorrelation", spearmanCorrelation)
      .build()
    metadata
  }

  def fromMetadata(metadata:Metadata): ColumnMetadata ={
    ColumnMetadata(
        metadata.getDouble("mean"),
        metadata.getDouble("max"),
        metadata.getDouble("min"),
        metadata.getDouble("per10"),
        metadata.getDouble("per50"),
        metadata.getDouble("per90"),
        metadata.getDouble("per99"),
        metadata.getLong("uniqCount"),
        metadata.getDouble("std"),
        metadata.getDouble("pirsonCorrelation"),
        metadata.getDouble("spearmanCorrelation"))
  }
}


case class DatasetMetadata(columnMetadatas: Map[String, ColumnMetadata]){

  def getColumnMetadata(columnName: String): ColumnMetadata ={
    columnMetadatas(columnName)
  }

 }

sealed class TransformerParam[T](override val parent: String,
                               override val name: String,
                               override val doc: String) extends Param[DatasetMetadata](parent, name, doc)
{
  override def jsonEncode(value: DatasetMetadata): String = {
    implicit val formats = {
      Serialization.formats(FullTypeHints(List(classOf[DatasetMetadata])))
    }
    compact(render(Extraction.decompose(value)))
  }
  override def jsonDecode(json: String): DatasetMetadata = {
    implicit val formats = {
      Serialization.formats(FullTypeHints(List(classOf[DatasetMetadata])))
    }
    parse(json).extract[DatasetMetadata]
  }
}

trait StatsTransformerParams extends StatsEstimatorParams {
    final val statMetadata = new TransformerParam[DatasetMetadata](this.uid, "metadata",
      "column statistical metadata")
    def setMetadata(metadata: DatasetMetadata): this.type = set(statMetadata, metadata)

}


class StatTransformer(override val uid: String)
  extends Model[StatTransformer] with StatsTransformerParams
  with DefaultParamsWritable
  with DefaultParamsReadable[StatTransformer]
{


  override def transform(dataset: Dataset[_]): DataFrame = {
    var df:DataFrame = dataset.toDF()

    for (column <- $(evaluatedColumns)) {
      val newColumn = df.col(column).as(column, $(statMetadata).getColumnMetadata(column).metadata())
      df = df.withColumn(column, newColumn)
    }
    df
  }

  override def copy(extra: ParamMap): StatTransformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

}
