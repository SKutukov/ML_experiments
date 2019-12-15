import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

object App {
  def main(args: Array[String]): Unit = {

    val spark:SparkSession = SparkSession
      .builder()
      .appName("basic example")
      .config("spark.master", "local")
      .getOrCreate()

    val df = spark.read
      .format("csv")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .load("/home/faraon/data/data.csv")

    val selected_df  = df.select(Seq("auditweights_ctr_gender", "auditweights_ctr_high", "auditweights_svd_prelaunch",
                                "liked")
      .map(
      c => col(c).cast("double")
    ): _*)

    var statTransformer = new StatTransformer("Stat")

    val transformed_df = statTransformer.transform(selected_df)
    transformed_df.schema.foreach{s => println(s"${s.name}, ${s.metadata.toString}")}

//    val assembler = new VectorAssembler()
//      .setInputCols(Array("auditweights_ctr_gender",
//            "auditweights_ctr_high", "auditweights_svd_prelaunch"))
//      .setOutputCol("features")
//
//    val features_df = assembler.transform(transformed_df)
//    features_df.show()
//    val lr = new LinearRegression()
//      .setLabelCol("liked")
//      .setMaxIter(10)
//      .setRegParam(0.3)
//      .setElasticNetParam(0.8)
//
//      Fit the model
//    val lrModel = lr.fit(features_df)
//
//      Print the coefficients and intercept for linear regression
//    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
//
//      Summarize the model over the training set and print out some metrics
//    val trainingSummary = lrModel.summary
//    println(s"numIterations: ${trainingSummary.totalIterations}")
//    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
//    trainingSummary.residuals.show()
//    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
//    println(s"r2: ${trainingSummary.r2}")

  }
}
