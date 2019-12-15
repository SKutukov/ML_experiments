import breeze.numerics.abs
import org.scalatest.FunSuite
import org.apache.spark.sql.SparkSession

class StatTransformerTest extends FunSuite{
  test("testTransformMetadata") {
    val eps = 0.0001
    val spark:SparkSession = SparkSession
      .builder()
      .appName("basic example")
      .config("spark.master", "local")
      .getOrCreate()

    val testDF = spark.createDataFrame(
      List(
        (2d, -0.5d, 3d),
        (0d, -0.5d, 2d),
        (-1d, -1d, 1d),
        (-1d, -2d, 0d)
      )
    ).toDF("x1","x2","target")

    val statEstimator = new  StatEstimator("Stat")
      .setTargetColumn("target")
      .setEvaluatedColumn(Array("x1","x2", "target"))

    val statTransformer = statEstimator.fit(testDF)

    val transformed_df = statTransformer.transform(testDF)
    val means = Map[String, Double]("x1" -> 0, "x2" -> -1, "target" -> 1.5)
    val maxs = Map[String, Double]("x1" -> 2, "x2" -> -0.5, "target" -> 3)
    val mins = Map[String, Double]("x1" -> -1, "x2" -> -2, "target" -> 0)
    val uniqCounts = Map[String, Long]("x1" -> 3, "x2" -> 3, "target" -> 4)
    val std = Map[String, Double]("x1" -> 1.414214, "x2" -> 0.7071068, "target" -> 1.290994)

    val pers10 = Map[String, Double]("x1" -> -1, "x2" -> -2, "target" -> 0)
    val pers50 = Map[String, Double]("x1" -> -1, "x2" -> -1, "target" -> 1)
    val pers90 = Map[String, Double]("x1" -> 2, "x2" ->  -0.5, "target" -> 3)
    val pers99 = Map[String, Double]("x1" -> 2, "x2" -> -0.5, "target" -> 3)


    val spirman = Map[String, Double]("x1" -> 0.9486833, "x2" -> 0.9486833, "target" -> 1)
    val pirson = Map[String, Double]("x1" -> 0.9128709, "x2" -> 0.9128709, "target" -> 1)

    transformed_df.schema.foreach {
        s => {
          assert(abs(s.metadata.getDouble("mean") - means(s.name)) < eps)
          assert(abs(s.metadata.getDouble("max") - maxs(s.name)) < eps)
          assert(abs(s.metadata.getDouble("min") - mins(s.name)) < eps)
          assert(s.metadata.getLong("uniqCount") === uniqCounts(s.name))
          assert(abs(s.metadata.getDouble("std") - std(s.name)) < eps)

          assert(abs(s.metadata.getDouble("per10") - pers10(s.name)) < eps)
          assert(abs(s.metadata.getDouble("per50") - pers50(s.name)) < eps)
          assert(abs(s.metadata.getDouble("per90") - pers90(s.name)) < eps)
          assert(abs(s.metadata.getDouble("per99") - pers99(s.name)) < eps)

          assert(abs(s.metadata.getDouble("spearmanCorrelation") - spirman(s.name)) < eps)
          assert(abs(s.metadata.getDouble("pirsonCorrelation") - pirson(s.name)) < eps)
        }
    }
  }

  test("testMeanStd") {
    val eps = 0.01
    val spark:SparkSession = SparkSession
      .builder()
      .appName("basic example")
      .config("spark.master", "local")
      .getOrCreate()

    val r = new scala.util.Random(100)
    val count = 10000
    val mean = 5
    val mean2 = 2
    val sd = 1
    val seq = for (i <- 1 to count) yield (r.nextGaussian(), r.nextGaussian() + mean, r.nextGaussian() + mean2)


    val testDF = spark.createDataFrame(
      seq
    ).toDF("x1","x2","y")

    val statEstimator = new  StatEstimator("Stat")
      .setTargetColumn("y")
      .setEvaluatedColumn(Array("x1","x2", "y"))

    val statTransformer = statEstimator.fit(testDF)

    val transformed_df = statTransformer.transform(testDF)
    val means = Map[String, Double]("x1" -> 0, "x2" -> 5, "y" -> 2)
    val std = Map[String, Double]("x1" -> 1, "x2" -> 1, "y" -> 1)

    transformed_df.schema.foreach {
      s => {
        assert(abs(s.metadata.getDouble("mean") - means(s.name)) < eps)
        assert(abs(s.metadata.getDouble("std") - std(s.name)) < eps)

      }
    }
  }

  test("testMinMax") {
    val eps = 0.01
    val spark:SparkSession = SparkSession
      .builder()
      .appName("basic example")
      .config("spark.master", "local")
      .getOrCreate()

    val r = new scala.util.Random(100)
    val count = 10000
    val mean = 5
    val mean2 = 2


    var seq = for (i <- 1 to count) yield (r.nextGaussian(), r.nextGaussian() + mean, r.nextGaussian() + mean2)
    seq = seq.updated(1, (100d, 200d, 150d))
    seq = seq.updated(2, (-150d, -200d, -150d))

    val testDF = spark.createDataFrame(
      seq
    ).toDF("x1","x2","y")

    val statEstimator = new  StatEstimator("Stat")
      .setTargetColumn("y")
      .setEvaluatedColumn(Array("x1","x2", "y"))

    val statTransformer = statEstimator.fit(testDF)

    val transformed_df = statTransformer.transform(testDF)
    val maxs = Map[String, Double]("x1" -> 100, "x2" -> 200, "y" -> 150)
    val mins = Map[String, Double]("x1" -> -150, "x2" -> -200, "y" -> -150)

    transformed_df.schema.foreach {
      s => {
            assert(abs(s.metadata.getDouble("max") - maxs(s.name)) < eps)
            assert(abs(s.metadata.getDouble("min") - mins(s.name)) < eps)

      }
    }
  }


    test("testUniqPer") {
      val eps = 0.01
      val spark:SparkSession = SparkSession
        .builder()
        .appName("basic example")
        .config("spark.master", "local")
        .getOrCreate()

      val count = 100

      var seq = for (i <- 1 to count) yield (i.toDouble, (i/10L).toDouble, (i/5L).toDouble)

      val testDF = spark.createDataFrame(
        seq
      ).toDF("x1","x2","y")

      val statEstimator = new  StatEstimator("Stat")
        .setTargetColumn("y")
        .setEvaluatedColumn(Array("x1","x2", "y"))

      val statTransformer = statEstimator.fit(testDF)

      val transformed_df = statTransformer.transform(testDF)

      val uniqCounts = Map[String, Long]("x1" -> 100, "x2" -> 11, "y" -> 21)

      val pers10 = Map[String, Double]("x1" -> 10d, "x2" -> 1d, "y" -> 2d)
      val pers50 = Map[String, Double]("x1" -> 50d, "x2" -> 5d, "y" -> 10d)
      val pers90 = Map[String, Double]("x1" -> 90d, "x2" ->  9d, "y" -> 18d)
      val pers99 = Map[String, Double]("x1" -> 99d, "x2" -> 9d, "y" -> 19d)

      transformed_df.schema.foreach {
        s => {
          assert(s.metadata.getLong("uniqCount") === uniqCounts(s.name))
          assert(s.metadata.getDouble("per10") === pers10(s.name))
          assert(s.metadata.getDouble("per50") === pers50(s.name))
          assert(s.metadata.getDouble("per90") === pers90(s.name))
          assert(s.metadata.getDouble("per99") === pers99(s.name))

        }
      }
    }

    test("testCorr") {
      val eps = 0.01
      val spark:SparkSession = SparkSession
        .builder()
        .appName("basic example")
        .config("spark.master", "local")
        .getOrCreate()

      val count = 100

      var seq = for (i <- 1 to count) yield (i.toDouble, (count - i).toDouble, i.toDouble)

      val testDF = spark.createDataFrame(
        seq
      ).toDF("x1","x2","y")

      val statEstimator = new  StatEstimator("Stat")
        .setTargetColumn("y")
        .setEvaluatedColumn(Array("x1","x2", "y"))

      val statTransformer = statEstimator.fit(testDF)

      val transformed_df = statTransformer.transform(testDF)
      val spirman = Map[String, Double]("x1" -> 1, "x2" -> -1, "y" -> 1)
      val pirson = Map[String, Double]("x1" -> 1, "x2" -> -1, "y" -> 1)

      transformed_df.schema.foreach {
        s => {

          assert(abs(s.metadata.getDouble("spearmanCorrelation") - spirman(s.name)) < eps)
          assert(abs(s.metadata.getDouble("pirsonCorrelation") - pirson(s.name)) < eps)

        }
      }
    }
}
