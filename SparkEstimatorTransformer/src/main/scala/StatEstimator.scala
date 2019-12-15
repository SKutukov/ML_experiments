import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.{Param, ParamMap, Params, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.{callUDF, col, corr, countDistinct, lit, max, mean, min, stddev, struct}
import org.apache.spark.sql.types.StructType


trait StatsEstimatorParams extends Params {
  final val targetColumn = new Param[String](this, "targetColumn", "The target column")
  final val evaluatedColumns = new StringArrayParam(this, "evaluatedColumns",
    "List of columns in which statistics will be evaluated")

  def setTargetColumn(column: String): this.type = set(targetColumn, column)
  def setEvaluatedColumn(columns: Array[String]): this.type = set(evaluatedColumns, columns)

}


class StatEstimator(override val uid: String) extends Estimator[StatTransformer]
  with StatsEstimatorParams{

  def this() = this(Identifiable.randomUID("statEstimator"))

  def percentile(col_name:String, per:Double): Column ={
    callUDF("percentile_approx", col(col_name), lit(per))
  }

  def spearman(col_name:String, target_col_name:String, df: DataFrame): Column ={
    lit(Statistics.corr(df.select(col_name).rdd.map(_.getDouble(0)),
                        df.select(target_col_name).rdd.map(_.getDouble(0)), "spearman"))
  }
  override def fit(dataset: Dataset[_]): StatTransformer = {

    val spark:SparkSession = SparkSession
      .builder()
      .appName("basic example")
      .config("spark.master", "local")
      .getOrCreate()

    val df:DataFrame = dataset.toDF()

    val res = df.select(struct($(evaluatedColumns).map(mean(_)): _*).as("means"),
                        struct($(evaluatedColumns).map(max(_)): _*).as("maxs"),
                        struct($(evaluatedColumns).map(min(_)): _*).as("mins"),
                        struct($(evaluatedColumns).map(stddev(_)): _*).as("stds"),
                        struct($(evaluatedColumns).map(countDistinct(_)): _*).as("countDistincts"),
                        struct($(evaluatedColumns).map(percentile(_, 0.1)): _*).as("percentile_approx_10"),
                        struct($(evaluatedColumns).map(percentile(_, 0.5)): _*).as("percentile_approx_50"),
                        struct($(evaluatedColumns).map(percentile(_, 0.9)): _*).as("percentile_approx_90"),
                        struct($(evaluatedColumns).map(percentile(_, 0.99)): _*).as("percentile_approx_99"),
                        struct($(evaluatedColumns).map(corr(_,$(targetColumn))): _*).as("pearson"),
                        struct($(evaluatedColumns).map(spearman(_, $(targetColumn), df)): _*).as("spearman"))

    var i = 0
    val col_result:Row = res.first()
    var columnMetadatas = Map[String, ColumnMetadata]()

    for (columnName <- $(evaluatedColumns)){

      val col_mean:Double = col_result.getAs[Row](0).getDouble(i)
      val col_max:Double = col_result.getAs[Row](1).getDouble(i)
      val col_min:Double = col_result.getAs[Row](2).getDouble(i)
      val col_std:Double = col_result.getAs[Row](3).getDouble(i)
      val col_uniqCount:Long = col_result.getAs[Row](4).getLong(i)
      val col_per10:Double = col_result.getAs[Row](5).getDouble(i)
      val col_per50:Double = col_result.getAs[Row](6).getDouble(i)
      val col_per90:Double = col_result.getAs[Row](7).getDouble(i)
      val col_per99:Double = col_result.getAs[Row](8).getDouble(i)
      val col_pirsonCorrelation:Double =  col_result.getAs[Row](9).getDouble(i)
      val col_spearmanCorrelation:Double = col_result.getAs[Row](10).getDouble(i)

      val columnMetadata = ColumnMetadata(col_mean,
        col_max,
        col_min,
        col_per10,
        col_per50,
        col_per90,
        col_per99,
        col_uniqCount,
        col_std,
        col_pirsonCorrelation,
        col_spearmanCorrelation
      )
      columnMetadatas += (columnName -> columnMetadata)
      i += 1

    }

    val datasetMetadata = DatasetMetadata(columnMetadatas)


    new StatTransformer(uid)
      .setTargetColumn($(targetColumn))
      .setEvaluatedColumn($(evaluatedColumns))
      .setMetadata(datasetMetadata)

  }

  override def copy(extra: ParamMap): Estimator[StatTransformer] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

}
