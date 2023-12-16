import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import java.io.*;

public class MyFSDataInputStream extends FSDataInputStream {
    public MyFSDataInputStream(InputStream in) {
        super(in);
    }
    /**
     * 实现按行读取
     * 每次读入一个字符，遇到"\n"结束，返回一行内容
     */
    public static String readline(BufferedReader br) throws IOException {
        char[] data = new char[1024];
        int read = -1;
        int off = 0;

        // 循环执行时，br 每次会从上一次读取结束的位置继续读取
        //因此该函数里，off 每次都从 0 开始
        while ( (read = br.read(data, off, 1)) != -1 ) {
            if (String.valueOf(data[off]).equals("\n") ) {
                off += 1;
                break;
            }
            off += 1;
        }
        if (off > 0) {
            //转换为字符串输出
            return String.valueOf(data);
        } else {
            return null;
        }
    }

    /**
     * 读取文件内容
     */
    public static void cat(Configuration conf, String remoteFilePath) throws IOException {
        //获取文件
        FileSystem fs = FileSystem.get(conf);
        Path remotePath = new Path(remoteFilePath);

        //打开读取文件
        FSDataInputStream in = fs.open(remotePath);
        BufferedReader br = new BufferedReader(new InputStreamReader(in));

        //按行输出文件内容
        String line = null;
        while ( (line = MyFSDataInputStream.readline(br)) != null ) {
            System.out.println(line);
        }

        //释放资源
        br.close();
        in.close();
        fs.close();
    }

    /**
     * 主函数
     */
    public static void main(String[] args) {
        Configuration conf = new Configuration();  //配置对象
        conf.set("fs.defaultFS","hdfs://43.138.121.194:9000"); //hadoop集群地址
        String path = "/user/hadoop/test/test1.txt"; // HDFS 文件路径

        try {
            MyFSDataInputStream.cat(conf, path);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}