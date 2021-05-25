import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.rdd._
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import java.lang.management.{ManagementFactory, ThreadMXBean}


object Main {

  type Embedding = (String, List[Double])
  type ParsedReview = (Integer, String, Double)

  org.apache.log4j.Logger getLogger "org" setLevel (org.apache.log4j.Level.WARN)
  org.apache.log4j.Logger getLogger "akka" setLevel (org.apache.log4j.Level.WARN)
  val spark = SparkSession.builder
    .appName("Sentiment")
    .master("local[5]")
    .getOrCreate

  spark.conf.set("spark.executor.memory", "4g")
  import spark.implicits._
  val reviewSchema = StructType( Array(
      StructField("reviewText", StringType, nullable = false),
      StructField("overall", DoubleType, nullable = false),
      StructField("summary", StringType, nullable = false)))

  def loadReviews(path: String): Dataset[ParsedReview] =
    spark.read.schema(reviewSchema).json(path).rdd.zipWithUniqueId.map[(Integer, String, Double)] {
        case (row, id) =>
          (id.toInt, s"${row getString 2} ${row getString 0}", row getDouble 1)}
      .toDS
      .withColumnRenamed("_1", "id").withColumnRenamed("_2", "text").withColumnRenamed("_3", "overall")
      .as[ParsedReview]

  def loadGlove(path: String): Dataset[Embedding] =
    spark.read.text(path).map { _ getString 0 split " " }
      .map(r => (r.head, r.tail.toList.map(_.toDouble)))
      .withColumnRenamed("_1", "word").withColumnRenamed("_2", "vec")
      .as[Embedding]

  def main(args: Array[String]) = {

    val DATA_PATH = ".../data"
    val glove: Dataset[Embedding] = loadGlove(s"${DATA_PATH}/glove.6B.50d.txt")
    val reviews: Dataset[ParsedReview] = loadReviews(s"${DATA_PATH}/Musical_Instruments_5.json")
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val tokenized: DataFrame = tokenizer.transform(reviews)

    def reviewsToEmbeddings(transformTokenized: DataFrame): Dataset[(Integer, String, Double)] = {
      transformTokenized.flatMap(a => a.getAs[WrappedArray[String]]("words").map(
        word =>Tuple3[Integer, String, Double](a.getAs("id"),word,a.getAs("overall"))))
        .withColumnRenamed("_1", "id").withColumnRenamed("_2", "word")
        .withColumnRenamed("_3", "overall")
        .as[(Integer, String, Double)]}
        
    val singleWords = reviewsToEmbeddings(tokenized)
    val joinedGlove: Dataset[(String, Integer, Double, Array[Double])] = singleWords.join(glove, "word")
        .withColumnRenamed("_1", "word").withColumnRenamed("_2", "id")
        .withColumnRenamed("_3", "overall").withColumnRenamed("_4", "vec")
        .as[(String, Integer, Double, Array[Double])]

    val dropColumnWord: DataFrame =joinedGlove.toDF.map(row =>Tuple3[Integer, Double, WrappedArray[Double]](
      row.getAs("id"),row.getAs("overall"),row.getAs("vec")))
        .withColumnRenamed("_1", "id").withColumnRenamed("_2", "overall").withColumnRenamed("_3", "vec")

    val addColumn1s: DataFrame = dropColumnWord.map(row =>Tuple4[Integer, Double, WrappedArray[Double], Integer](
      row.getAs("id"),row.getAs("overall"),row.getAs("vec"),1))
      .withColumnRenamed("_1", "id").withColumnRenamed("_2", "overall")
      .withColumnRenamed("_3", "vec").withColumnRenamed("_4", "1s")

    def sumTheVectors(first: WrappedArray[Double],secound: WrappedArray[Double]): WrappedArray[Double] = {
      first.zip(secound).map { case (x, y) => x + y }}

    def eachRowDivideVectorByCount(vec: WrappedArray[Double],count: Integer): WrappedArray[Double] = {
      vec.map(_ / count)}

    val reduceWithKey: Dataset[(Integer, Double, Array[Double])] = addColumn1s.rdd
      .map(row =>(row.getAs("id"): Integer,
        (row.getAs("1s"): Integer,row.getAs[WrappedArray[Double]]("vec"),row.getAs[Double]("overall"))))
          .reduceByKey((x, y) => (x._1 + y._1, sumTheVectors(x._2, y._2), x._3))
          .map(row =>(row._1, row._2._3, eachRowDivideVectorByCount(row._2._2, row._2._1)))
           .toDS
            .withColumnRenamed("_1", "id").withColumnRenamed("_2", "overall").withColumnRenamed("_3", "vec")
              .as[(Integer, Double, Array[Double])]

    val properDtaForMLPC: Dataset[(Integer, Integer, Vector)] =reduceWithKey.map(row =>(
      row._1,
      if (row._2 < 3) 0 else if (row._2 == 5) 2 else 1,Vectors.dense(row._3)))
        .withColumnRenamed("_1", "id").withColumnRenamed("_2", "label").withColumnRenamed("_3", "features")
        .as[(Integer, Integer, Vector)]

    val splits = properDtaForMLPC.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val train = splits(0)
    val test = splits(1)
    val layers = Array[Int](50, 4, 4, 3)
    val trainer = new MultilayerPerceptronClassifier()
    .setMaxIter(50).setLayers(layers).setBlockSize(128).setSeed(1234L)

    def measureTime[T](f: => T): (T, (Double, Double)) = {
      val bean: ThreadMXBean = ManagementFactory.getThreadMXBean()
      def getCpuTime = if (bean.isCurrentThreadCpuTimeSupported()) bean.getCurrentThreadCpuTime() else 0
      val start_cpu = getCpuTime
      val start_wall = System.nanoTime
      val result = f
      val end_wall = System.nanoTime
      val end_cpu = getCpuTime
      val t_cpu = (end_cpu - start_cpu) / 1000000000.0
      val t_wall = (end_wall - start_wall) / 1000000000.0
      (result, (t_cpu, t_wall)) 
  }

    val accuracies = (0 to 9).map(foldnr => {
    val train = properDtaForMLPC.rdd.zipWithIndex.filter(
      (t) => {t._2 % 10 != foldnr}
    ).map(t => t._1).toDS.withColumnRenamed("_1", "id")
    .withColumnRenamed("_2", "label").withColumnRenamed("_3", "features")
    .as[(Integer, Integer, Vector)]

    val test = properDtaForMLPC.rdd.zipWithIndex.filter(
      (t) => {t._2 % 10 == foldnr}
    ).map(t => t._1).toDS.withColumnRenamed("_1", "id")
    .withColumnRenamed("_2", "label").withColumnRenamed("_3", "features")
    .as[(Integer, Integer, Vector)] 

    val (model, (cputime, walltime)) = measureTime(trainer.fit(train))
    println(s"Training on fold ${foldnr} took ${cputime} seconds in cputime and ${walltime} seconds in wall clock time")
    
    val model10F = trainer.fit(train)
    val result10F = model10F.transform(test)
    val predictionAndLabels10F = result10F.select("prediction", "label")
    val evaluator10F = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    evaluator10F.evaluate(predictionAndLabels10F)
  })
  
    def mean(xs: Seq[Double]): Option[Double] =
    if (xs.isEmpty) None
    else Some(xs.sum / xs.length)

    def variance (xs: Seq[Double]): Option[Double] = 
    mean(xs).flatMap(m => mean(xs.map(x => math.pow(x - m, 2))))
 
    println("Accuracies across folds :")
    println(accuracies)
    println(s"variance of accuracy observed in 10-Fold Cross-Validation = ${variance(accuracies.toSeq)}")

    spark.stop
  }
}
