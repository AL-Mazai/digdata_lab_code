package k_means.center;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

// 随机为data分配簇
// 输出簇编号方便reduce计算
// setup中读取k大小
public class CenterRandomMapper extends Mapper<Object, Text, Text, Text> {

    private int k;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        // 读取k值
        Configuration configuration = context.getConfiguration();
        k = configuration.getInt("cluster.k", 3);
        System.out.println("k:" + k);
    }

    @Override
    protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        // 随机分配簇
        int index = (int) (Math.random() * k);
        context.write(new Text(Integer.toString(index)), value);
    }
}
