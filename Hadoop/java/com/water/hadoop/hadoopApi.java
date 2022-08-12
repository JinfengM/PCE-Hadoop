package com.water.hadoop;


import java.io.File;
import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.HColumnDescriptor;
import org.apache.hadoop.hbase.HTableDescriptor;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.Admin;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.client.ConnectionFactory;

public class hadoopApi {
    public hadoopApi() {
    }

    public static String _getFileName(String file) {
        try {
            File file1 = new File(file);
            return file1.getName();
        } catch (Exception e) {
            e.printStackTrace();
            return "";
        }
    }

    public static boolean uploadfile_static(String dst, String src, Configuration cfg) {
        try {
            File fs0 = new File(src);
            if (!fs0.exists()) {
                return false;
            } else {
                FileSystem fs = FileSystem.get(cfg);
                fs.copyFromLocalFile(false, true, new Path(src), new Path(dst));
                fs.close();
                return true;
            }
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    public static boolean uploadfile_static(String dst, String src) {
        try {
            Configuration cfg = new Configuration();
            File fs0 = new File(src);
            if (!fs0.exists()) {
                return false;
            } else {
                FileSystem fs = FileSystem.get(cfg);
                fs.copyFromLocalFile(false, true, new Path(src), new Path(dst));
                fs.close();
                return true;
            }
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    public static void uploadfolder_static(String dstfolder, String srcfolder, Configuration cfg) {
    	File root = new File(srcfolder);
        File[] files = root.listFiles();

        for(File file:files)
        {
            if(file.isDirectory())
            {
                System.out.println("====================== directory");
                String dstfolderChild = String.format("%s/%s", dstfolder, _getFileName(file.getAbsolutePath()));
                uploadfolder_static(dstfolderChild, file.getAbsolutePath(), cfg);
            }
            else
            {
                String src = file.getAbsolutePath();
                String dst = String.format("%s/%s", dstfolder, _getFileName(src));
                uploadfile_static(dst, src, cfg);
            }
        }
    }

    public static void uploadfolder_static(String dstfolder, String srcfolder) {
    	File root = new File(srcfolder);
        File[] files = root.listFiles();

        for(File file:files)
        {
            if(file.isDirectory())
            {
                System.out.println("====================== directory");
                dstfolder = String.format("%s/%s", dstfolder, _getFileName(file.getAbsolutePath()));
                uploadfolder_static(dstfolder, file.getAbsolutePath());
            }
            else
            {
                String src = file.getAbsolutePath();
                String dst = String.format("%s/%s", dstfolder, _getFileName(src));
                uploadfile_static(dst, src);
            }
        }
    }

    public static void downloadfile_static(String src, String dst, Configuration cfg) {
        try {
            FileSystem fs = FileSystem.get(cfg);
            fs.copyToLocalFile(false, new Path(src), new Path(dst));
            fs.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static void downloadfolder_static(String srcfolder, String dstfolder, Configuration cfg) {
        try {
            File dstDir = new File(dstfolder);
            if (!dstDir.exists()) {
                dstDir.mkdirs();
            } else {
                dstDir.delete();
            }

            FileSystem fs = FileSystem.get(cfg);
            FileStatus[] srcFileStatus = fs.listStatus(new Path(srcfolder));

            for(int i = 0; i < srcFileStatus.length; ++i) {
                String srcfile = srcFileStatus[i].getPath().toString();
                String srcName = srcFileStatus[i].getPath().getName();
                String dstfile = String.format("%s/%s", dstfolder, srcName);
                downloadfile_static(dstfile, srcfile, cfg);
            }

            fs.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static void CreateTable(String tablename, String[] cols) {
        try {
            Configuration conf = new Configuration(true);
            conf.set("mapreduce.app-submission.cross-platform", "true");
            Configuration hBaseConf = HBaseConfiguration.create();
            Connection hBaseConn = ConnectionFactory.createConnection(hBaseConf);
            Admin hAdmin = hBaseConn.getAdmin();
            TableName tbName = TableName.valueOf(tablename);
            if (hAdmin.tableExists(tbName)) {
                return;
            }

            HTableDescriptor hTableDescriptor = new HTableDescriptor(tbName);  
            
            for(String col:cols){  
                HColumnDescriptor hColumnDescriptor = new HColumnDescriptor(col);  
                hTableDescriptor.addFamily(hColumnDescriptor);  
            }  
            hAdmin.createTable(hTableDescriptor);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) {
    }
}
