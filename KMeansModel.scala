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

import scala.collection.JavaConverters._

import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.pmml.PMMLExportable
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
// 训练完成后 生成KMeansModel 模型包含中心点向量
/**
 * A clustering model for K-means. Each point belongs to the cluster with the closest center.
 */
@Since("0.8.0")
class KMeansModel @Since("1.1.0") (@Since("1.0.0") val clusterCenters: Array[Vector])
  extends Saveable with Serializable with PMMLExportable {

  /**
   * A Java-friendly constructor that takes an Iterable of Vectors.
   */
  @Since("1.4.0")
  def this(centers: java.lang.Iterable[Vector]) = this(centers.asScala.toArray)   //构造函数 asScala转化器方法 将一个java收集对应到Scala的集合中去

  /**
   * Total number of clusters.
   */
  @Since("0.8.0")
  def k: Int = clusterCenters.length

  /**
   * Returns the cluster index that a given point belongs to.
   */
  @Since("0.8.0")    			// 预测函数 根据训练的模型也就是得到的中心点向量 预测输入的样本点向量  计算样本点里最近的中心点 然后输出最近中心点标号 
  def predict(point: Vector): Int = {
    KMeans.findClosest(clusterCentersWithNorm, new VectorWithNorm(point))._1
  }

  /**
   * Maps given points to their cluster indices.
   */
  @Since("1.0.0")
  def predict(points: RDD[Vector]): RDD[Int] = {  // 重写函数 参数不同 得到样本点
    val centersWithNorm = clusterCentersWithNorm   //将clusterCenters Array[Vector]类型转换为 Iterable[VectorWithNorm]
    val bcCentersWithNorm = points.context.broadcast(centersWithNorm)   // 广播变量
    points.map(p => KMeans.findClosest(bcCentersWithNorm.value, new VectorWithNorm(p))._1) 
  }

  /**
   * Maps given points to their cluster indices.
   */
  @Since("1.0.0")
  def predict(points: JavaRDD[Vector]): JavaRDD[java.lang.Integer] =
    predict(points.rdd).toJavaRDD().asInstanceOf[JavaRDD[java.lang.Integer]]

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  @Since("0.8.0")
  def computeCost(data: RDD[Vector]): Double = {   //计算样本点到最近中心点的平方距离之和
    val centersWithNorm = clusterCentersWithNorm
    val bcCentersWithNorm = data.context.broadcast(centersWithNorm)
    data.map(p => KMeans.pointCost(bcCentersWithNorm.value, new VectorWithNorm(p))).sum()
  }

  private def clusterCentersWithNorm: Iterable[VectorWithNorm] =   
    clusterCenters.map(new VectorWithNorm(_))

  @Since("1.4.0")
  override def save(sc: SparkContext, path: String): Unit = {     // 保存模型
    KMeansModel.SaveLoadV1_0.save(sc, this, path)
  }

  override protected def formatVersion: String = "1.0"
}

@Since("1.4.0")
object KMeansModel extends Loader[KMeansModel] {

  @Since("1.4.0")
  override def load(sc: SparkContext, path: String): KMeansModel = {
    KMeansModel.SaveLoadV1_0.load(sc, path)
  }

  private case class Cluster(id: Int, point: Vector)

  private object Cluster {
    def apply(r: Row): Cluster = { //工厂方法 不用new 
      Cluster(r.getInt(0), r.getAs[Vector](1)) // 得到每一行 然后转化为Cluster类 id和point
    }
  }

  private[clustering]
  object SaveLoadV1_0 {

    private val thisFormatVersion = "1.0"

    private[clustering]
    val thisClassName = "org.apache.spark.mllib.clustering.KMeansModel"

    def save(sc: SparkContext, model: KMeansModel, path: String): Unit = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()   //SparkSession 是Spark2.0之后新引入的新的切入点 为DataSet和Dataframe提供的API 
																		 //SparkSession内部封装了sparkContext，所以计算实际上是由sparkContext完成的。
      val metadata = compact(render(
        ("class" -> thisClassName) ~ ("version" -> thisFormatVersion) ~ ("k" -> model.k)))  //compact函数返回字符串
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))   //parallelize 创建RDD函数  然后保存元数据
      val dataRDD = sc.parallelize(model.clusterCenters.zipWithIndex).map { case (point, id) =>    //将模型中心点标号 Array((A,0), (B,1), (R,2), (D,3), (F,4))
        Cluster(id, point)                                                   //保存为Cluster类
      }
      spark.createDataFrame(dataRDD).write.parquet(Loader.dataPath(path))    //DataFrame是一个以命名列方式组织的分布式数据集 将数据写为Parquet文件列式存储格式
    }

    def load(sc: SparkContext, path: String): KMeansModel = {  //将中心点保存为parquet文件
      implicit val formats = DefaultFormats
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      val (className, formatVersion, metadata) = Loader.loadMetadata(sc, path) //读出元数据
      assert(className == thisClassName)  //assert 判断条件错误后 程序直接终止
      assert(formatVersion == thisFormatVersion)
      val k = (metadata \ "k").extract[Int]
      val centroids = spark.read.parquet(Loader.dataPath(path)) //读取数据文件
      Loader.checkSchema[Cluster](centroids.schema)  	//得到parquet 文件中schema结构 每个schema的结构是这样的：
														//根叫做message，message包含多个fields。每个field包含三个属性：repetition, type, name。
														//repetition可以是以下三种：required（出现1次），optional（出现0次或者1次），repeated（出现0次或者多次）。
														//type可以是一个group或者一个primitive类型。
      val localCentroids = centroids.rdd.map(Cluster.apply).collect()  //
      assert(k == localCentroids.length)
      new KMeansModel(localCentroids.sortBy(_.id).map(_.point))  //返回KMeansModel类 
    }
  }
}