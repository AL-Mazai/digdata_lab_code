import util.JdbcUtil;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public class LinkMysql {
    public static void main(String[] args) throws ClassNotFoundException, SQLException {
        //获取数据库连接
        Connection connection = JdbcUtil.getConnection();
        //需要执行的sql语句
        String sql = "select English from student where Name = ?";
        //获取预处理对象，并给参数赋值
        PreparedStatement statement = connection.prepareCall(sql);
        statement.setString(1, "张三");
        //执行sql语句
        ResultSet resultSet = statement.executeQuery();  //executeQuery：执行并返回结果集
        while (resultSet.next()) {
            System.out.println("张三English成绩为:" + resultSet.getInt("English"));
        }
        //关闭jdbc连接
        resultSet.close();
        statement.close();
        connection.close();
    }
}
