package k_means;

import k_means.center.CenterRandomAdapter;
import k_means.cluster.KmeansAdapter;
import k_means.util.DataUtil;

// kmeans主方法
// 主要就是循环调用直到次数或者停机状态
// 每次循环判断下停机
// data为逗号连接
public class KmeansRun {
    public static void main(String[] args) {

        //聚类数
        int k = 10;

        String centerPath = DataUtil.HDFS_OUTPUT + "centers"; //存储中心点
        String newCenterPath = DataUtil.HDFS_OUTPUT + "new_centers"; //存储新的中心点
        String dataPath = DataUtil.HDFS_INPUT + "data.txt"; //数据集
        String clusterResultPath = DataUtil.HDFS_OUTPUT + "kmeans_cluster_result"; //结果

        // 初始化随机中心点
        CenterRandomAdapter.createRandomCenter(dataPath, centerPath, k);
        // 默认500次，中途停退出
        for (int i = 0; i < 500; i++) {
            KmeansAdapter.start(dataPath, centerPath, newCenterPath);
            if (KmeansAdapter.checkStop(centerPath, newCenterPath))
                break;
        }

        KmeansAdapter.createClusterResult(dataPath, centerPath, clusterResultPath);
    }
}
