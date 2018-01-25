/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.clustering

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.annotation.Since
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.clustering.{KMeans => NewKMeans}
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.BLAS.{axpy, scal}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

/**
 * K-means clustering with a k-means++ like initialization mode
 * (the k-means|| algorithm by Bahmani et al).
 *// kmeans++算法选择初始seeds的基本思想就是：初始的聚类中心之间的相互距离要尽可能的远。
//从输入的数据点集合中随机选择一个点作为第一个聚类中心
//对于数据集中的每一个点x，计算它与最近聚类中心(指已选择的聚类中心)的距离D(x)
//选择一个新的数据点作为新的聚类中心，选择的原则是：D(x)较大的点，被选取作为聚类中心的概率较大
//重复2和3直到k个聚类中心被选出来
 * This is an iterative algorithm that will make multiple passes over the data, so any RDDs given
 * to it should be cached by the user.
 */
@Since("0.8.0")
class KMeans private (
    private var k: Int, 
    private var maxIterations: Int,
    private var initializationMode: String,
    private var initializationSteps: Int,//初始化步长 为kmeans++设计
    private var epsilon: Double,
    private var seed: Long) extends Serializable with Logging {

  /**
   * Constructs a KMeans instance with default parameters: {k: 2, maxIterations: 20,
   * initializationMode: "k-means||", initializationSteps: 2, epsilon: 1e-4, seed: random}.
   */
  @Since("0.8.0")
  def this() = this(2, 20, KMeans.K_MEANS_PARALLEL, 2, 1e-4, Utils.random.nextLong())  // 构造函数 设置默认值

  /**
   * Number of clusters to create (k).
   *
   * @note It is possible for fewer than k clusters to
   * be returned, for example, if there are fewer than k distinct points to cluster.
   */
  @Since("1.4.0")
  def getK: Int = k 

  /**
   * Set the number of clusters to create (k).
   *
   * @note It is possible for fewer than k clusters to
   * be returned, for example, if there are fewer than k distinct points to cluster. Default: 2.
   */
  @Since("0.8.0")
  def setK(k: Int): this.type = {//其返回值 还是这个类 可以实现调用自身
    require(k > 0,               //require scala中的函数 第一个参数为boolean类型 如果第一个参数为false 则抛出异常并输出第二个参数信息
      s"Number of clusters must be positive but got ${k}")
    this.k = k
    this
  }

  /**
   * Maximum number of iterations allowed.
   */
  @Since("1.4.0")
  def getMaxIterations: Int = maxIterations

  /**
   * Set maximum number of iterations allowed. Default: 20.
   */
  @Since("0.8.0")
  def setMaxIterations(maxIterations: Int): this.type = {
    require(maxIterations >= 0,
      s"Maximum of iterations must be nonnegative but got ${maxIterations}")
    this.maxIterations = maxIterations
    this
  }

  /**
   * The initialization algorithm. This can be either "random" or "k-means||".
   */
  @Since("1.4.0")
  def getInitializationMode: String = initializationMode

  /**
   * Set the initialization algorithm. This can be either "random" to choose random points as
   * initial cluster centers, or "k-means||" to use a parallel variant of k-means++
   * (Bahmani et al., Scalable K-Means++, VLDB 2012). Default: k-means||.
   */
  @Since("0.8.0")
  def setInitializationMode(initializationMode: String): this.type = {
    KMeans.validateInitMode(initializationMode)     //在validateInitMode 函数中做了个模式匹配 确保输入的字符串为‘random’ 、‘k-means||’
    this.initializationMode = initializationMode
    this
  }

  /**
   * This function has no effect since Spark 2.0.0.
   */
  @Since("1.4.0")
  @deprecated("This has no effect and always returns 1", "2.1.0")
  def getRuns: Int = {
    logWarning("Getting number of runs has no effect since Spark 2.0.0.")
    1
  }

  /**
   * This function has no effect since Spark 2.0.0.
   */
  @Since("0.8.0")
  @deprecated("This has no effect", "2.1.0")
  def setRuns(runs: Int): this.type = {
    logWarning("Setting number of runs has no effect since Spark 2.0.0.")
    this
  }

  /**
   * Number of steps for the k-means|| initialization mode
   */
  @Since("1.4.0")
  def getInitializationSteps: Int = initializationSteps

  /**
   * Set the number of steps for the k-means|| initialization mode. This is an advanced
   * setting -- the default of 2 is almost always enough. Default: 2.
   */
  @Since("0.8.0")
  def setInitializationSteps(initializationSteps: Int): this.type = {
    require(initializationSteps > 0,
      s"Number of initialization steps must be positive but got ${initializationSteps}")
    this.initializationSteps = initializationSteps
    this
  }

  /**
   * The distance threshold within which we've consider centers to have converged.
   */
  @Since("1.4.0")
  def getEpsilon: Double = epsilon

  /**
   * Set the distance threshold within which we've consider centers to have converged.
   * If all centers move less than this Euclidean distance, we stop iterating one run.
   */
  @Since("0.8.0")
  def setEpsilon(epsilon: Double): this.type = {
    require(epsilon >= 0,
      s"Distance threshold must be nonnegative but got ${epsilon}")
    this.epsilon = epsilon
    this
  }

  /**
   * The random seed for cluster initialization.
   */
  @Since("1.4.0")
  def getSeed: Long = seed

  /**
   * Set the random seed for cluster initialization.
   */
  @Since("1.4.0")
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  // Initial cluster centers can be provided as a KMeansModel object rather than using the
  // random or k-means|| initializationMode
  private var initialModel: Option[KMeansModel] = None

  /**
   * Set the initial starting point, bypassing the random initialization or k-means||
   * The condition model.k == this.k must be met, failure results
   * in an IllegalArgumentException.
   */
  @Since("1.4.0")
  def setInitialModel(model: KMeansModel): this.type = {   //本函数可以直接绕过 以上两种初始化中心点的方式 直接指定 KMeansModel的参数是一个中心点的数组 
    require(model.k == k, "mismatched cluster count")
    initialModel = Some(model)
    this
  }

  /**
   * Train a K-means model on the given set of points; `data` should be cached for high
   * performance, because this is an iterative algorithm.
   */
  @Since("0.8.0")
  def run(data: RDD[Vector]): KMeansModel = { //输入为数据RDD 返回KmeansModel 类型
    run(data, None)
  }

  private[spark] def run(   //[spark] 是指 本函数只能在spark包下运行
      data: RDD[Vector],
      instr: Option[Instrumentation[NewKMeans]]): KMeansModel = {

    if (data.getStorageLevel == StorageLevel.NONE) {  // 如果输入数据RDD 没有缓存 输出个警报 
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }
	  //计算范数平方
    // Compute squared norms and cache them.
    val norms = data.map(Vectors.norm(_, 2.0)) //调用map函数对每一行数据求L2范数
    norms.persist() //求的数据缓存
    val zippedData = data.zip(norms).map { case (v, norm) =>  // zip 拉链函数 将两个参数进行连接 相应位置上的元素组成一个pair数组 如果其中一个参数元素比较长，那么多余的参数会被删掉 
      new VectorWithNorm(v, norm)							  // 在这里相当于把原数据与所有L2范数相连 之后将他们转化为VectorWithNorm类 为了以后调用方便
    }
    val model = runAlgorithm(zippedData, instr)        //将转化后的数据传入进行计算
    norms.unpersist()                                    // 计算完成后将L2范数在内存中去除          

    // Warn at the end of the run as well, for increased visibility.
    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data was not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }
    model  // 返回值
  }

  /**
   * Implementation of K-Means algorithm.
   */
  private def runAlgorithm(  //计算距离函数
      data: RDD[VectorWithNorm], // 得到输入数据RDD所需类型为VectorWithNorm 以上已经转换
      instr: Option[Instrumentation[NewKMeans]]): KMeansModel = { //返回KMeansModel类

    val sc = data.sparkContext   //得到spark的接口函数

    val initStartTime = System.nanoTime() //计算当前系统时间

    val centers = initialModel match {   //initialModel为option类型 如果有值则引用值 如果没有则会引用None
      case Some(kMeansCenters) =>   //Some为Option的子类  上边定义的函数setInitialMode 绕过 random 和kmeans++方法 自定义的KMeansModel类型 
        kMeansCenters.clusterCenters.map(new VectorWithNorm(_))  //将自定义的clusterCenters 转化为VectorWithNorm类型
      case None =>
        if (initializationMode == KMeans.RANDOM) {  //如果没有自定义 则为None 判断是random 还是kmeans++
          initRandom(data) //335行
        } else {
          initKMeansParallel(data)//351
        }
    }
    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9  //计算初始化中心点时间
    logInfo(f"Initialization with $initializationMode took $initTimeInSeconds%.3f seconds.") //记录日志

    var converged = false
    var cost = 0.0
    var iteration = 0

    val iterationStartTime = System.nanoTime()

    instr.foreach(_.logNumFeatures(centers.head.vector.size))

    // Execute iterations of Lloyd's algorithm until converged   //执行 弗洛伊德算法 直到收敛为止 寻找给定的加权图中顶点间最短路径的算法
    while (iteration < maxIterations && !converged) {
      val costAccum = sc.doubleAccumulator //创建一个累加器 名字叫costAccum
      val bcCenters = sc.broadcast(centers)//广播centers变量 名字叫bcCenters

      // Find the sum and count of points mapping to each center
      val totalContribs = data.mapPartitions { points =>
        val thisCenters = bcCenters.value 
        val dims = thisCenters.head.vector.size //得到数据的维数

        val sums = Array.fill(thisCenters.length)(Vectors.zeros(dims))   //填充数组 填充中心点个数个dims维的零向量   ((0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0))
        val counts = Array.fill(thisCenters.length)(0L) //用0填充中心点个数长的数组  (0,0,0,0,0)

        points.foreach { point => //foreach 和map类似 不过没有返回值
          val (bestCenter, cost) = KMeans.findClosest(thisCenters, point) //577行
          costAccum.add(cost) // 累加器  距离总和 
          val sum = sums(bestCenter)  //点到最近距离的簇ID  sum指向sums中第几个向量
          axpy(1.0, point.vector, sum) //采用了BLAS线性代数运算库   axpy常数乘以向量加另一个向量  样本点放入sum
          counts(bestCenter) += 1    //第几个簇 置1  相当于（0，0，0，1，0，0）
         }

        counts.indices.filter(counts(_) > 0).map(j => (j, (sums(j), counts(j)))).iterator   //indices 返回一个range 从0到counts.length Range（0,1,2,3，5...） map之后（0，（（0,0,0），（0,0,0,1,0））） 之后用转化为一个迭代器
      }.reduceByKey { case ((sum1, count1), (sum2, count2)) =>     //reduceByKey 必须对键值对操作  合并具有相同键的值
        axpy(1.0, sum2, sum1)
        (sum1, count1 + count2)
      }.collectAsMap()//用于Pair RDD，最终返回Map类型 Map(A ->B ,  C->D)

      bcCenters.destroy(blocking = false)

      // Update the cluster centers and costs
      converged = true
      totalContribs.foreach { case (j, (sum, count)) =>
        scal(1.0 / count, sum)                    //  scal x=a*x    sum =1/count * sum
        val newCenter = new VectorWithNorm(sum)
        if (converged && KMeans.fastSquaredDistance(newCenter, centers(j)) > epsilon * epsilon) {  //如果 得到的中心点减去以前的每个中心点 小于 阈值 的收敛
          converged = false
        }
        centers(j) = newCenter
      }

      cost = costAccum.value
      iteration += 1
    }

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9  //计算迭代时间
    logInfo(f"Iterations took $iterationTimeInSeconds%.3f seconds.")

    if (iteration == maxIterations) {
      logInfo(s"KMeans reached the max number of iterations: $maxIterations.")
    } else {
      logInfo(s"KMeans converged in $iteration iterations.")
    }

    logInfo(s"The cost is $cost.")

    new KMeansModel(centers.map(_.vector))  // 得到K个中心点 返回KMeansModel类 
  }

  /**
   * Initialize a set of cluster centers at random.
   */
  private def initRandom(data: RDD[VectorWithNorm]): Array[VectorWithNorm] = {
    // Select without replacement; may still produce duplicates if the data has < k distinct
    // points, so deduplicate the centroids to match the behavior of k-means|| in the same situation
    data.takeSample(false, k, new XORShiftRandom(this.seed).nextInt()) // spark函数采样， 第一个参数为boolean类型判断是否需要替换， 第二个参数取K个 ，第三个为随机种子 设置种子保证得到的随机数相同 返回dataRDD的子集
																		//XORShiftRandom 是spark中的实现的一个类 比调用java.util.Random 快
      .map(_.vector).distinct.map(new VectorWithNorm(_))                //对得到的中心点去重 
  }

  /**
   * Initialize a set of cluster centers using the k-means|| algorithm by Bahmani et al.
   * (Bahmani et al., Scalable K-Means++, VLDB 2012). This is a variant of k-means++ that tries
   * to find dissimilar cluster centers by starting with a random center and then doing
   * passes where more centers are chosen with probability proportional to their squared distance
   * to the current cluster set. It results in a provable approximation to an optimal clustering.
   *
   * The original paper can be found at http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf.
   */
 *// kmeans++算法选择初始seeds的基本思想就是：初始的聚类中心之间的相互距离要尽可能的远。
//从输入的数据点集合中随机选择一个点作为第一个聚类中心
//对于数据集中的每一个点x，计算它与最近聚类中心(指已选择的聚类中心)的距离D(x)
//选择一个新的数据点作为新的聚类中心，选择的原则是：D(x)较大的点，被选取作为聚类中心的概率较大
//重复2和3直到k个聚类中心被选出来
   
  private[clustering] def initKMeansParallel(data: RDD[VectorWithNorm]): Array[VectorWithNorm] = {
    // Initialize empty centers and point costs.
    var costs = data.map(_ => Double.PositiveInfinity) // 生成data数据集中样例个数个正无穷大数

    // Initialize the first center to a random point.
    val seed = new XORShiftRandom(this.seed).nextInt() //
    val sample = data.takeSample(false, 1, seed) 得到第一个聚类中心点
    // Could be empty if data is empty; fail with a better message early:
    require(sample.nonEmpty, s"No samples available from $data")

    val centers = ArrayBuffer[VectorWithNorm]() //创建个ArrayBuffer可扩展的数组 不知道K值 类型为VectorWithNorm
    var newCenters = Seq(sample.head.toDense) //将得到的中心点 转化为dense vector
    centers ++= newCenters //将转化后的dense vector 加入到centers可扩展数组中

    // On each step, sample 2 * k points on average with probability proportional
    // to their squared distance from the centers. Note that only distances between points
    // and new centers are computed in each iteration.
	// 在每次迭代中 抽样2*k个样本进行计算
    // 每次迭代中计算样本点与中心点之间的距离
    var step = 0
    var bcNewCentersList = ArrayBuffer[Broadcast[_]]()   //创建可扩展数组 类型为广播变量类型 因为是分布式计算 所以其他节点需要得到中心点的值进行本地计算 广播变量会发送一个只读值给所有节点
    while (step < initializationSteps) {
      val bcNewCenters = data.context.broadcast(newCenters)  // 广播newCenters的值 广播变量的名字为bcNewCenters
      bcNewCentersList += bcNewCenters   //将广播变量加入到bcNewCentersList 可扩展数组中
      val preCosts = costs
      costs = data.zip(preCosts).map { case (point, cost) =>  // 用zip拉链函数 使data与costs结合 使用map对每一行进行操作
        math.min(KMeans.pointCost(bcNewCenters.value, point), cost)   //  调用findClosest 函数计算 里中心点最近的距离 每次迭代去最小值 567行
      }.persist(StorageLevel.MEMORY_AND_DISK)  
	//对于每个点，我们都计算其和最近的一个“种子点”的距离D(x)并保存在一个数组里，然后把这些距离加起来得到Sum(D(x))。
	//然后，再取一个随机值，用权重的方式来取计算下一个“种子点”。这个算法的实现是，先取一个能落在Sum(D(x))中的随机值Random，然后用Random -= D(x)，直到其<=0，此时的点就是下一个“种子点”。
	//重复2和3直到k个聚类中心被选出来
      val sumCosts = costs.sum()

      bcNewCenters.unpersist(blocking = false)// blocking 默认为ture  false时 不用等到所有都删除在执行下条语句
      preCosts.unpersist(blocking = false)

      val chosen = data.zip(costs).mapPartitionsWithIndex { (index, pointCosts) =>  //map函数类似，只不过映射函数的参数由RDD中的每一个元素变成了RDD中每一个分区的迭代器 如果在映射的过程中需要频繁创建额外的对象，使用mapPartitions要比map高效的过。
        val rand = new XORShiftRandom(seed ^ (step << 16) ^ index)
        pointCosts.filter { case (_, c) => rand.nextDouble() < 2.0 * c * k / sumCosts }.map(_._1)
      }.collect()
      newCenters = chosen.map(_.toDense)
      centers ++= newCenters
      step += 1
    }

    costs.unpersist(blocking = false)
    bcNewCentersList.foreach(_.destroy(false))//销毁与此广播变量相关的所有数据和元数据。谨慎使用 一旦广播变量被破坏，就不能再使用了

    val distinctCenters = centers.map(_.vector).distinct.map(new VectorWithNorm(_)) //去重

    if (distinctCenters.size <= k) {
      distinctCenters.toArray
    } else {
      // Finally, we might have a set of more than k distinct candidate centers; weight each
      // candidate by the number of points in the dataset mapping to it and run a local k-means++
      // on the weighted centers to pick k of them
      val bcCenters = data.context.broadcast(distinctCenters)
      val countMap = data.map(KMeans.findClosest(bcCenters.value, _)._1).countByValue()   //countByValue统计出集合中每个元素的个数[值，个数]

      bcCenters.destroy(blocking = false)

      val myWeights = distinctCenters.indices.map(countMap.getOrElse(_, 0L).toDouble).toArray  //indices生成该序列所有索引的范围 getOrElse得到值如果没有置零
      LocalKMeans.kMeansPlusPlus(0, distinctCenters.toArray, myWeights, k, 30)    //LocalKMeans用于本地运行k-均值
    }
  }
}


/**
 * Top-level methods for calling K-means clustering.
 */
@Since("0.8.0")
object KMeans {  //KMeans类的伴生对象 其中的方法函数可以直接调用

  // Initialization mode names
  @Since("0.8.0")
  val RANDOM = "random"
  @Since("0.8.0")
  val K_MEANS_PARALLEL = "k-means||"

  /**
   * Trains a k-means model using the given set of parameters.
   *
   * @param data Training points as an `RDD` of `Vector` types.
   * @param k Number of clusters to create.
   * @param maxIterations Maximum number of iterations allowed.
   * @param initializationMode The initialization algorithm. This can either be "random" or
   *                           "k-means||". (default: "k-means||")
   * @param seed Random seed for cluster initialization. Default is to generate seed based
   *             on system time.
   */
  @Since("2.1.0")
  def train(
      data: RDD[Vector],
      k: Int,
      maxIterations: Int,
      initializationMode: String,
      seed: Long): KMeansModel = {
    new KMeans().setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .setSeed(seed)
      .run(data)
  }

  /**
   * Trains a k-means model using the given set of parameters.
   *
   * @param data Training points as an `RDD` of `Vector` types.
   * @param k Number of clusters to create.
   * @param maxIterations Maximum number of iterations allowed.
   * @param initializationMode The initialization algorithm. This can either be "random" or
   *                           "k-means||". (default: "k-means||")
   */
  @Since("2.1.0")
  def train(
      data: RDD[Vector],
      k: Int,
      maxIterations: Int,
      initializationMode: String): KMeansModel = {
    new KMeans().setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .run(data)
  }

  /**
   * Trains a k-means model using the given set of parameters.
   *
   * @param data Training points as an `RDD` of `Vector` types.
   * @param k Number of clusters to create.
   * @param maxIterations Maximum number of iterations allowed.
   * @param runs This param has no effect since Spark 2.0.0.
   * @param initializationMode The initialization algorithm. This can either be "random" or
   *                           "k-means||". (default: "k-means||")
   * @param seed Random seed for cluster initialization. Default is to generate seed based
   *             on system time.
   */
  @Since("1.3.0")
  @deprecated("Use train method without 'runs'", "2.1.0")
  def train(
      data: RDD[Vector],
      k: Int,
      maxIterations: Int,
      runs: Int,
      initializationMode: String,
      seed: Long): KMeansModel = {
    new KMeans().setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .setSeed(seed)
      .run(data)
  }

  /**
   * Trains a k-means model using the given set of parameters.
   *
   * @param data Training points as an `RDD` of `Vector` types.
   * @param k Number of clusters to create.
   * @param maxIterations Maximum number of iterations allowed.
   * @param runs This param has no effect since Spark 2.0.0.
   * @param initializationMode The initialization algorithm. This can either be "random" or
   *                           "k-means||". (default: "k-means||")
   */
  @Since("0.8.0")
  @deprecated("Use train method without 'runs'", "2.1.0")
  def train(
      data: RDD[Vector],
      k: Int,
      maxIterations: Int,
      runs: Int,
      initializationMode: String): KMeansModel = {
    new KMeans().setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .run(data)
  }

  /**
   * Trains a k-means model using specified parameters and the default values for unspecified.
   */
  @Since("0.8.0")
  def train(
      data: RDD[Vector],
      k: Int,
      maxIterations: Int): KMeansModel = {
    new KMeans().setK(k)
      .setMaxIterations(maxIterations)
      .run(data)
  }

  /**
   * Trains a k-means model using specified parameters and the default values for unspecified.
   */
  @Since("0.8.0")
  @deprecated("Use train method without 'runs'", "2.1.0")
  def train(
      data: RDD[Vector],
      k: Int,
      maxIterations: Int,
      runs: Int): KMeansModel = {
    new KMeans().setK(k)
      .setMaxIterations(maxIterations)
      .run(data)
  }

  /**
   * Returns the index of the closest center to the given point, as well as the squared distance.
   Mlib的kmeans采用一种快速查找 计算距离的方法。
    lowerBoundOfSqDist距离公式，假设中心点center是(a1,b1)，需要计算的点point是(a2,b2)，那么lowerBoundOfSqDist是： 
    lowerBoundOfSqDist =（根号下a1的平方+b1的平方  -  根号下a2的平方+b2的平方）的平方
	EuclideanDist = （a1-a2）的平方 +（b1-b2）的平方  
	在查找最短距离的时候无需计算开方，因为只需要计算出开方里面的式子就可以进行比较了， 只需计算center 和opint的L2范数
    可轻易证明上面两式的第一式将会小于等于第二式，因此在进行距离比较的时候，先计算很容易计算的lowerBoundOfSqDist，
	如果lowerBoundOfSqDist都不小于之前计算得到的最小距离bestDistance，那真正的欧式距离也不可能小于bestDistance了，
	因此这种情况下就不需要去计算欧式距离，省去很多计算工作。
	如果lowerBoundOfSqDist小于了bestDistance，则进行距离的计算，调用fastSquaredDistance，这个方法将调用MLUtils.scala里面的fastSquaredDistance方法，计算真正的欧式距离，	
	
   */
  private[mllib] def findClosest(  //传入两个餐数据 一个是中心点数据格式为VectorWithNorm 另一个是和中心点计算距离的点
      centers: TraversableOnce[VectorWithNorm],
      point: VectorWithNorm): (Int, Double) = { //返回簇ID 和距离
    var bestDistance = Double.PositiveInfinity  //先定义一个double型极大数
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      // Since `\|a - b\| \geq |\|a\| - \|b\||`, we can use this lower bound to avoid unnecessary
      // distance computation.
      var lowerBoundOfSqDist = center.norm - point.norm  //计算center 和opint的L2范数
      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
      if (lowerBoundOfSqDist < bestDistance) {
        val distance: Double = fastSquaredDistance(center, point)
        if (distance < bestDistance) {
          bestDistance = distance
          bestIndex = i
        }
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  /**
   * Returns the K-means cost of a given point against the given cluster centers.
   */
  private[mllib] def pointCost(
      centers: TraversableOnce[VectorWithNorm],
      point: VectorWithNorm): Double =
    findClosest(centers, point)._2

  /**
   * Returns the squared Euclidean distance between two vectors computed by
   * [[org.apache.spark.mllib.util.MLUtils#fastSquaredDistance]].
   */
  private[clustering] def fastSquaredDistance( 
      v1: VectorWithNorm,
      v2: VectorWithNorm): Double = {
    MLUtils.fastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm) //计算欧式距离
  }

  private[spark] def validateInitMode(initMode: String): Boolean = {
    initMode match {
      case KMeans.RANDOM => true
      case KMeans.K_MEANS_PARALLEL => true
      case _ => false
    }
  }
}

/**
 * A vector with its norm for fast distance computation.
 *
 * @see [[org.apache.spark.mllib.clustering.KMeans#fastSquaredDistance]]
 */
private[clustering]
class VectorWithNorm(val vector: Vector, val norm: Double) extends Serializable {

  def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))

  def this(array: Array[Double]) = this(Vectors.dense(array))

  /** Converts the vector to a dense vector. */
  def toDense: VectorWithNorm = new VectorWithNorm(Vectors.dense(vector.toArray), norm)
}