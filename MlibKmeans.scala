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
    //spark�Ľӿ� ����һЩ��������
    val conf = new SparkConf().setAppName("MllibKmeans")//.setMaster("local")
    val sc = new SparkContext(conf)
    //������־��ʾ�ȼ� ������ֻ��ʾ��WARN ERROR
    Logger.getRootLogger.setLevel(Level.WARN)
    // ����
    val numClusters = args(0).toInt //�صĸ���
    val numIterations = args(1).toInt//��������
    val input =args(2)//��������·��
    val output=args(3)//���ģ��·��

    //�ж����·������ �еĻ�ɾ��
    val path = Path(output)
    if(path.exists) {path.deleteRecursively()}

    val start = System.currentTimeMillis() //��¼��ʼʱ��

    val data = sc.textFile(input)  //�������� ����RDD

    //�������� ����RDD  ��������ΪCSV��ʽ�����Զ��ŷָ� ��ÿ��Ԫ��ת��ΪDouble�� ��Ϊ�������Զ�RDD���浽�ڴ�
    val parseddata = data.map(s => Vectors.dense(s.split(",").map(_.toDouble))).cache()

    val initMode = "random" //�������ĵ��ʼ��ģʽ random ������ k-means||
    // ����Mlilib��KMeans ���� �������
    val model = new KMeans().
      setK(numClusters).
      setInitializationMode(initMode).
      setMaxIterations(numIterations).
      run(parseddata)
    //����ѵ����ģ��
    val saveMode =model.save(sc ,output )
    //��ԭ���ݽ���Ԥ��
    val pre = model.predict(parseddata)
    //����  ����ƽ�����
    val WSSSE = model.computeCost(parseddata)
    //��¼����ʱ��
    val end = System.currentTimeMillis()

    //��� ÿ�����ĵ�
    model.clusterCenters.foreach(x=> println("cluster center"+ x ))
    println("Within Set Sum of Squared Errors = "+ WSSSE)
    println("hua fei shijian :"+ (end - start))
    

    println("------------------------------------------------")


  }

}