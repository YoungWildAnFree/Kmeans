import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import scala.reflect.io.Path
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

/**
  * Created by new on 2017/4/14.
  */
object MllibKmeans {
  def main(args: Array[String]): Unit = {
    //spark的接口 配置一些基本参数
    val conf = new SparkConf().setAppName("MllibKmeans")//.setMaster("local")
    val sc = new SparkContext(conf)
    //设置日志显示等级 这里是只显示是WARN ERROR
    Logger.getRootLogger.setLevel(Level.WARN)
    // 参数
    val numClusters = args(0).toInt //簇的个数
    val numIterations = args(1).toInt//迭代次数
    val input =args(2)//输入数据路径
    val output=args(3)//输出模型路径

    //判断输出路径有无 有的话删除
    val path = Path(output)
    if(path.exists) {path.deleteRecursively()}

    val start = System.currentTimeMillis() //记录开始时间

    val data = sc.textFile(input)  //读入数据 创建RDD

    //读入数据 创建RDD  本次数据为CSV格式所以以逗号分隔 将每个元素转化为Double型 因为迭代所以对RDD缓存到内存
    val parseddata = data.map(s => Vectors.dense(s.split(",").map(_.toDouble))).cache()

    val initMode = "random" //设置中心点初始化模式 random 或者是 k-means||
    // 调用Mlilib中KMeans 函数 传入参数
    val model = new KMeans().
      setK(numClusters).
      setInitializationMode(initMode).
      setMaxIterations(numIterations).
      run(parseddata)
    //保存训练的模型
    val saveMode =model.save(sc ,output )
    //对原数据进行预测
    val pre = model.predict(parseddata)
    //评估  计算平方差和
    val WSSSE = model.computeCost(parseddata)
    //记录结束时间
    val end = System.currentTimeMillis()

    //输出 每个中心点
    model.clusterCenters.foreach(x=> println("cluster center"+ x ))
    println("Within Set Sum of Squared Errors = "+ WSSSE)
    println("hua fei shijian :"+ (end - start))
    

    println("------------------------------------------------")


  }

}