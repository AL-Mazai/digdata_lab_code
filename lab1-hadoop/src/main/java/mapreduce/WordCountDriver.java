package mapreduce;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCountDriver {

    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        //1、获取job的配置信息
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf);

        //2、设置jar的加载路径
        job.setJarByClass(WordCountDriver.class);

        //3、分别设置Mapper和Reducer类
        job.setMapperClass(WordCountMapper.class);
        job.setReducerClass(WordCountReducer.class);

        //4、设置map的输出类型
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        //5、设置最终输出的键值类型
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        //6、设置输入输出路径
        FileInputFormat.setInputPaths(job, new Path("/usr/WordCountTest.txt"));
        FileOutputFormat.setOutputPath(job, new Path("/wordCount"));

        //7、提交任务
        boolean flag = job.waitForCompletion(true);
        System.out.println("flag : " + flag);
    }
}
