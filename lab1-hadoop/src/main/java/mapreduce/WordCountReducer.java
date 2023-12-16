package mapreduce;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * Text 		-  输入的键（即Mapper阶段输出的键）
 * IntWritable 	- 输入的值(个数)(即Mapper阶段输出的值)
 * Text 		- 输出的键
 * IntWritable 	- 输出的值
 */
public class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

    //text为输入的键，value为输入的内容
    @Override
    protected void reduce(Text text, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        //统计键对应的个数
        int sum = 0;
        for (IntWritable value : values) {
            sum = sum + value.get();
        }

        //设置reducer的输出
        IntWritable v = new IntWritable();
        v.set(sum);
        context.write(text, v);
    }
}
