package com.water.hadoop;

import com.alibaba.fastjson.JSONArray;
import com.water.oper._base.SubCellInfo;
import com.water.swat._base._baserch;
import com.water.swat.calibrate.ObjectiveFunction;
import com.water.swat.calibrate._calswatparam;
import com.water.swat.calibrate.calibrate_param;
import com.water.swat.calibrate.editswatfile;
import com.water.swat.output.outfile_rch;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.URI;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;
import java.util.function.ToDoubleFunction;
import java.util.function.ToLongFunction;
import org.apache.commons.httpclient.DefaultHttpMethodRetryHandler;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.HttpException;
import org.apache.commons.httpclient.methods.PostMethod;
import org.apache.commons.httpclient.methods.RequestEntity;
import org.apache.commons.httpclient.methods.StringRequestEntity;
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
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Table;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

public class rmrcalswat {
	
    public rmrcalswat() {
    }
    
    public static String sendMsgToServer(String url) {
        String resultMsg = null;
        HttpClient httpClient = new HttpClient();
        PostMethod postMethod = new PostMethod(url);
        postMethod.getParams().setParameter("http.method.retry-handler", new DefaultHttpMethodRetryHandler());
        int ctimeout = httpClient.getHttpConnectionManager().getParams().getConnectionTimeout();
        int stimeout = httpClient.getHttpConnectionManager().getParams().getSoTimeout();
        httpClient.getHttpConnectionManager().getParams().setConnectionTimeout(5000);
        httpClient.getHttpConnectionManager().getParams().setSoTimeout(5000);

        try {
            int statusCode = httpClient.executeMethod(postMethod);
            if (statusCode != 200) {
                return null;
            }
        } catch (HttpException e1) {
            ;
        } catch (IOException e) {
            ;
        } finally {
            postMethod.releaseConnection();
        }

        return (String)resultMsg;
    }

    public static void main(String[] args) throws Exception {
    	String swatLocalFolder = "D:\\work\\calibrate\\meijiang";
    	String paramFile = "bo_flowout.txt";
    	String pythonLocalParamFile = "D:\\work\\calibrate\\meijiang\\param\\13.txt";
    	
    	calibrate_param cps = new calibrate_param();
        cps.loadInfo(swatLocalFolder, paramFile);
        editswatfile esf = new editswatfile();           
        List<_calswatparam> params = esf.editparams(pythonLocalParamFile);
        
        for(int i = 0; i < params.size(); ++i) {
            _calswatparam p = params.get(i);
            p.name = ((_calswatparam)cps.params.get(p.pid)).name;
            p.suffix = ((_calswatparam)cps.params.get(p.pid)).suffix;
            params.set(i, p);
        }

        try {      
//            esf.editSwats(swatLocalFolder, params);
        } catch (Exception e8) {
        }
//
//    	
//    	
//    	List<_calswatparam> params = editSwatFiles(swatLocalFolder, pythonLocalParamFile);
    	
//        System.out.println(Double.parseDouble("1"));
//        calibrate_param cps = new calibrate_param();
//        cps.loadInfo(swatLocalFolder, paramFile);
//        editswatfile esf = new editswatfile();
//       
//        List<_calswatparam> params = esf.editparams(pythonparamFile);
//
//        for(int i = 0; i < params.size(); ++i) {
//            _calswatparam p = params.get(i);
//            p.name = ((_calswatparam)cps.params.get(p.pid)).name;
//            p.suffix = ((_calswatparam)cps.params.get(p.pid)).suffix;
//            params.set(i, p);
//        }
//        System.out.println(Double.parseDouble("2"));
    }

    public static class IntSumReducer
       extends Reducer<Text,IntWritable,Text,IntWritable> {
    	
    	class ThreadSubMap extends Thread {
    	 	   private Thread t;
    	 	   private String threadName;
    	 	   private boolean bStop = false;
    	 	   private long threadcurTime = 0;
    	 	   private long threadlastTime = 0;
    	 	   private long runtimes = 2880;
    	 	   private Context context = null;
    	 	   ThreadSubMap( String name, Context ctxt) {
    	 	      threadName = name;
    	 	      context = ctxt;
    	 	   }
    	 	   
    	 	   public void _Stop()
    	 	   {
    	 		   bStop = true;
    	 	   }
    	 	   
    	 	   @Override
    	 	   public void run() {
    	 	      try {
    	 	    	  
    	 	         while(!bStop && runtimes >=0)
    	 	         {
    	 	        	runtimes--;
    	 				threadcurTime = System.currentTimeMillis(); 	        	
    	 				if(threadcurTime-threadlastTime >= 180000)	//3min发送一次心跳
    	 				{
    	 					progress(context);
    	 					threadlastTime = threadcurTime;
    	 					logout("send a heart context to namenode;");
    	 				}
    	 				Thread.sleep(180000);
    	 	         }
    	 	      }catch (InterruptedException e) {
    	 	         logout("ThreadSunMap show eroor~");
    	 	      }
    	 	   }
    	 	   
    	 	   @Override
    	 	   public void start () 
    	 	   {
    	 	      if (t == null) 
    	 	      {
    	 	         t = new Thread (this, threadName);
    	 	         t.start ();
    	 	      }
    	 	   }
    	 	}
        	    	
            public HData hdata = new HData();
            public String engine = "swat";
            public String hdfsCalibratePath = "/calibrate";        
            public String paramIndex = null;
            public String swatName = null;
            public String calibType = null;
            public String pythonUrl = null;
            public String paramFile = null;
            public String obsFile = null;
            
            public String gTableName = null;
            public String calibalgo = "";
            long lasttime = 0L;
            long curtime = 0L;
            public Configuration hConf = new Configuration();
            public Configuration hBaseConf = null;
            private static final IntWritable one = new IntWritable(1);
            private Text word = new Text();

            public void InitConfig() {
                if (hConf == null) {
                    System.out.println("InitConfig().................");
                    ClearConfig();
                    hConf = new Configuration();
                    System.out.println("hConf=" + hConf);
                }
            }

            public void ClearConfig() {
                if (hConf != null) {
                    hConf.clear();
                }
            }

            public Connection getHBaseConn() {
                if (hBaseConf == null) {
                    hBaseConf = HBaseConfiguration.create();
                }

                try {
                    Connection hBaseConn = ConnectionFactory.createConnection(hBaseConf);
                    return hBaseConn;
                } catch (IOException e) {
                    e.printStackTrace();
                    return null;
                }
            }

            public Admin getHBaseAdmin() {
                try {
                    Connection hBaseConn = getHBaseConn();
                    Admin hAdmin = hBaseConn.getAdmin();
                    return hAdmin;
                } catch (IOException e) {
                    e.printStackTrace();
                    return null;
                }
            }

            public void ClearHBaseConfig() throws IOException {
                if (hBaseConf != null) {
                    hBaseConf.clear();
                }

            }

            public void CreateTable(String tablename, String[] cols) throws IOException {
            	Admin hAdmin = getHBaseAdmin();
                TableName tbName = TableName.valueOf(tablename);
                //是否删除表?或者清空数据?deleteTable
                if(hAdmin.tableExists(tbName)){  
                    System.out.println("table is exists!");  
                }else {          	
                    HTableDescriptor hTableDescriptor = new HTableDescriptor(tbName);  
                    
                    for(String col:cols){  
                        HColumnDescriptor hColumnDescriptor = new HColumnDescriptor(col);  
                        hTableDescriptor.addFamily(hColumnDescriptor);  
                    }  
                    hAdmin.createTable(hTableDescriptor);              
                }  
            }

            public void deleteTable(String tableName, Admin hAdmin) throws IOException {
                TableName tn = TableName.valueOf(tableName);
                if (hAdmin.tableExists(tn)) {
                    hAdmin.disableTable(tn);
                    hAdmin.deleteTable(tn);
                }

            }

            public void deleteTable(String tableName) throws IOException {
                Admin hAdmin = getHBaseAdmin();
                TableName tn = TableName.valueOf(tableName);
                if (hAdmin.tableExists(tn)) {
                    hAdmin.disableTable(tn);
                    hAdmin.deleteTable(tn);
                }

            }

            public void run(Context context) throws IOException, InterruptedException {		
        		super.run(context);
        	}

            public void downloadfile(String dst, String src, Context context) {
                try {
                    if (hConf == null) {
                        InitConfig();
                    }

                    FileSystem fs = FileSystem.newInstance(URI.create(src), hConf);
                    fs.copyToLocalFile(false, new Path(src), new Path(dst));
                    fs.close();
                    ping(context);
                } catch (IOException e) {
                    logout("下载文件异常 - file" + src + e.toString());
                    e.printStackTrace();
                }

            }

            public void downloadfolder(String dstfolder, String srcfolder, Context context) {
                try {
                    if (hConf == null) {
                        InitConfig();
                    }

                    FileSystem fs = FileSystem.newInstance(URI.create(srcfolder), hConf);
                    FileStatus[] srcFileStatus = fs.listStatus(new Path(srcfolder));

                    for(int i = 0; i < srcFileStatus.length; ++i) {
                        String srcfile = srcFileStatus[i].getPath().toString();
                        String srcName = srcFileStatus[i].getPath().getName();
                        String dstfile = String.format("%s/%s", dstfolder, srcName);
                        if (srcFileStatus[i].isDirectory()) {
                            downloadfolder(dstfile, srcfile, context);
                        } else {
                            downloadfile(dstfile, srcfile, context);
                        }
                    }

                    fs.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }

            }

            public boolean uploadfile(String dst, String src, Mapper<Object, Text, Text, IntWritable>.Context context) {
                try {
                    Configuration cfg = new Configuration();
                    File fs0 = new File(src);
                    if (!fs0.exists()) {
                        logout("上传文件异常(文件不存在)-srcfile: " + src + "   -dstfile: " + dst);
                        return false;
                    } else {
                        FileSystem fs = FileSystem.newInstance(URI.create(src), cfg);
                        fs.copyFromLocalFile(false, true, new Path(src), new Path(dst));
                        fs.close();
                        return true;
                    }
                } catch (IOException e7) {
                    logout("上传文件异常-srcfile: " + src + "   -dstfile: " + dst);
                    e7.printStackTrace();
                    return false;
                }
            }

            public void uploadfolder(String dstfolder, String srcfolder, String absfolder, Mapper<Object, Text, Text, IntWritable>.Context context) {
            	File root = new File(srcfolder);            
        		File[] files = root.listFiles();
        		for(File file:files)
        		{
        			if(file.isDirectory())
        			{
        				uploadfolder(dstfolder, file.getAbsolutePath(), absfolder, context);
        			}
        			else
        		    {
        				String src = file.getAbsolutePath();
        				String dst = src.replaceAll(absfolder, dstfolder);
        				uploadfile(dst, src, context);
        		    }			
        		}
            }

            public void logout(String log) {
                BufferedWriter out = null;

                try {
                    SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                    String file = null;
                    file = String.format("/home/hadoop/worklx/log-calib-swat-%s-%s.txt", swatName, paramIndex);
                    out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file, true)));
                    String msg = String.format("[%s]%s\r\n", df.format(new Date()), log);
                    out.write(msg);
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    try {
                        out.close();
                    } catch (IOException e13) {
                        e13.printStackTrace();
                    }
                }
            }

            public void ping(Context context) {
                curtime = System.currentTimeMillis();
                if (curtime - lasttime > 100000L) {
                    context.progress();
                    lasttime = curtime;
                }
            }

            public void progress(Context context) {
                context.progress();
            }

            protected void cleanup(Context context) throws IOException, InterruptedException {
                if (paramIndex != null && swatName != null) {
                    String sfolder = String.format("/home/hadoop/worklx/calibrate/swat-%s-%s-%s", 
                    						swatName, calibalgo, paramIndex);
                    logout("cleanup");

                    while(true) {
                        File dstDir = new File(sfolder);
                        if (!dstDir.exists()) {
                            return;
                        }
                        try {
                            String delaa = String.format("rm -rf %s", sfolder);
                            Runtime.getRuntime().exec(new String[]{"/bin/sh", "-c", delaa});
                            logout("clean up delete tmp folder");
                            Thread.sleep(60000L);
                            progress(context);
                        } catch (IOException e5) {
                            e5.printStackTrace();
                            Thread.sleep(60000L);
                            logout("clean up deleting tmp folder...............................");
                            progress(context);
                        }
                    }
                }
            }

            public void createFolder(String folder) {
                try {
                    logout(folder);
                    File file0 = new File(folder);
                    if (!file0.isDirectory()) {
                        file0.mkdir();
                        Thread.sleep(10000L);
                    }
                } catch (Exception e3) {
                    logout(e3.toString());
                }
            }

            protected void setup(Context context) throws IOException, InterruptedException 
            {
            }

            public boolean uploadHBase() throws IOException {
                try {
                    Connection hBaseConn = getHBaseConn();
                    Table table = hBaseConn.getTable(TableName.valueOf(gTableName));
                    String strKey = String.format("%s-sim-%s-%05d", calibType, calibalgo, hdata.paramIndex);
                    
                    List<Put> putList = new ArrayList();
                    Put p = new Put(Bytes.toBytes(strKey));
                    
                    String json = null;
                    
                    json = JSONArray.toJSONString(hdata.paramIndex);
                    p.addColumn(Bytes.toBytes("info"), 
                    		Bytes.toBytes("paramindex"), 
                    		Bytes.toBytes(json));
                    
                    json = JSONArray.toJSONString(hdata.paramvalues);
                    p.addColumn(Bytes.toBytes("info"), 
                    		Bytes.toBytes("paramvalue"), 
                    		Bytes.toBytes(json));
                    
                    json = JSONArray.toJSONString(hdata.nses);
                    p.addColumn(Bytes.toBytes("info"), 
                    		Bytes.toBytes("nse"), 
                    		Bytes.toBytes(json));
                    
                    json = JSONArray.toJSONString(hdata.flowout_sim);
                    p.addColumn(Bytes.toBytes("info"), 
                    		Bytes.toBytes("flowout"), 
                    		Bytes.toBytes(json));
                    putList.add(p);
                    
                    if (paramIndex.equals("0")) {
                        logout("paramIndex.equals int");
                        strKey = String.format("%s-obs-%s-00000", calibType, calibalgo);
                        Put p0 = new Put(Bytes.toBytes(strKey));
                        
                        json = JSONArray.toJSONString(hdata.paramIndexs);
                        p0.addColumn(Bytes.toBytes("info"), 
                        		Bytes.toBytes("paramindexs"), 
                        		Bytes.toBytes(json));
                        
                        json = JSONArray.toJSONString(hdata.reachs);
                        p0.addColumn(Bytes.toBytes("info"),
                        		Bytes.toBytes("reachs"), 
                        		Bytes.toBytes(json));
                        
                        json = JSONArray.toJSONString(hdata.dates);
                        p0.addColumn(Bytes.toBytes("info"), 
                        		Bytes.toBytes("dates"), 
                        		Bytes.toBytes(json));
                        
                        json = JSONArray.toJSONString(hdata.flowout_obs);
                        p0.addColumn(Bytes.toBytes("info"), 
                        		Bytes.toBytes("flowout"), 
                        		Bytes.toBytes(json));
                        putList.add(p0);
                    }

                    table.put(putList);
                    table.close();
                    hBaseConn.close();
                    return true;
                } catch (IOException e8) {
                    e8.printStackTrace();
                    logout(e8.toString());
                    return false;
                }
            }

            public String sendPythonResultParams(String paramJson) {
                System.out.println("params:==============================");
                System.out.println(paramJson);
                String url = pythonUrl;
                String resultMsg = null;
                HttpClient httpClient = new HttpClient();
                PostMethod postMethod = new PostMethod(url);

                try {
                    postMethod.getParams().setParameter("http.method.retry-handler", new DefaultHttpMethodRetryHandler());
                    postMethod.getParams().setContentCharset("UTF-8");
                    RequestEntity se = new StringRequestEntity(paramJson, "application/json", "UTF-8");
                    postMethod.setRequestEntity(se);
                    int ctimeout = httpClient.getHttpConnectionManager().getParams().getConnectionTimeout();
                    int stimeout = httpClient.getHttpConnectionManager().getParams().getSoTimeout();
                    httpClient.getHttpConnectionManager().getParams().setConnectionTimeout(60000);
                    httpClient.getHttpConnectionManager().getParams().setSoTimeout(60000);
                    int statusCode = httpClient.executeMethod(postMethod);
                    System.out.println("statuscode= " + statusCode);
                    if (statusCode != 200) {
                        return null;
                    }

                    byte[] responseBody = postMethod.getResponseBody();
                    resultMsg = new String(responseBody);
                    System.out.println("result:==============================");
                    System.out.println(resultMsg);
                } catch (HttpException e15) {
                    System.out.println("Please check your provided http address!");
                    e15.printStackTrace();
                } catch (IOException e16) {
                    ;
                } finally {
                    postMethod.releaseConnection();
                }

                return resultMsg;
            }

            public void runSwat(String swatLocalFolder, Context context) throws IOException, InterruptedException {
                int nResult = 0;
                String line = "";
    			try{
    				String shfile = String.format("cd %s;chmod +x -R %s;./lswat.exe", swatLocalFolder, swatLocalFolder);				
    			    Process ps = Runtime.getRuntime().exec(new String[] { "/bin/sh","-c", shfile});
    			    BufferedReader br = new BufferedReader(new InputStreamReader(ps.getInputStream()));
    			    while((line = br.readLine())!=null ){
    			    	logout(line);
    			    	if(line.indexOf("successfully")>=0)
    			    	{
    					   nResult++;
    					}
    			    	ping(context);
    			    }
                } catch (IOException e8) {
                    line = String.format("rm -rf %s", swatLocalFolder);
                    Runtime.getRuntime().exec(new String[]{"/bin/sh", "-c", line});
                    logout("run_swat  Exception , :  " + line);
                    logout("run_swat failed" + e8.toString());
                    e8.printStackTrace();
                    setSceneState(-1);
                    super.setup(context);
                    return;
                }

                if (nResult < 1) {
                    logout("calculate -  run_swat failed");
                    Thread.sleep(20000L);
                    String delaa = String.format("rm -rf %s", swatLocalFolder);
                    Runtime.getRuntime().exec(new String[]{"/bin/sh", "-c", delaa});
                    setSceneState(-1);
                    super.setup(context);
                } else {
                    logout("calculate result:-  run_swat success");
                }
            }

            public List<Double> resultDatas(String swatLocalFolder, String calibtype, Context context) {
                String obsfile = String.format("%s/param/%s", swatLocalFolder, obsFile);
                logout("obsfile: " + obsfile);
                outfile_rch rch = new outfile_rch();
                List<List<_baserch>> obsRchsList = rch.loadInfo_obs(obsfile, calibtype);
                if (obsRchsList != null && obsRchsList.size() > 0) {
                    logout("obsRchsList-size: " + obsRchsList.size());
                    int size = 0;
                    List<Double> NSEs = new ArrayList();
                    double NSETotal = 0.0D;
                    int pdex = 0;
                    hdata.dates = new long[obsRchsList.size()][];
                    hdata.reachs = new int[obsRchsList.size()];
                    hdata.flowout_obs = new double[obsRchsList.size()][];
                    hdata.flowout_sim = new double[obsRchsList.size()][];
                    hdata.nses = new double[obsRchsList.size()];
                    Iterator var13 = obsRchsList.iterator();

                    for(List<_baserch> obsRchs : obsRchsList) 
                    {   
                        if (obsRchs.size() <= 0) 
                        	continue;
                        int rchIndex = obsRchs.get(0).rch;
                        List<_baserch> simRchs = rch.loadInfo_sim(swatLocalFolder, calibtype, rchIndex);
                        if (simRchs.size() <= 0)
                        	continue;
                        
                        List<ArrayList<Double>> lss = rch.CompareRchs(simRchs, obsRchs);
                        if (lss == null) 
                        {
                            logout("simList and obsList error; rchIndex = " + rchIndex);
                        } 
                        else 
                        {
                            logout("simList and obsList good; rchIndex = " + obsRchs.get(0).rch);
                            ArrayList<Double> simList = lss.get(0);
                            ArrayList<Double> obsList = lss.get(1);
                            if (simList.size() != obsList.size()) 
                            {
                                logout("simList.size() != obsList.size(), rchIndex= " + rchIndex);
                                continue;
                            } 
                            else 
                            {
                                ping(context);
                                logout("start cut_swat.. rchIndex = " + rchIndex);

                                try {
                                    double NSETmp = ObjectiveFunction.ens(simList, obsList);
                                    logout(String.format("rchIndex = %d, NSE = %f ", rchIndex, NSETmp));
                                    NSEs.add(NSETmp);
                                    hdata.nses[pdex] = NSETmp;
                                    NSETotal += Math.abs(NSETmp);
                                    ++size;
                                } catch (Exception e) {
                                    ;
                                }
                                
                                logout("end cut_swat.. rchIndex = " + rchIndex);
                                hdata.reachs[pdex] = rchIndex;
                                ArrayList<Double> dateList = (ArrayList)lss.get(2);
                                logout("dateList.size() = " + dateList.size());
                                hdata.dates[pdex] = _getDateList(dateList);
                                logout("hdata.dates.length = " + hdata.dates.length);
                                hdata.flowout_obs[pdex] = obsList.stream().mapToDouble(Double::doubleValue).toArray();
                                logout("hdata.flowout_obs.length = " + hdata.dates.length);
                                hdata.flowout_sim[pdex] = simList.stream().mapToDouble(Double::doubleValue).toArray();
                                logout("hdata.flowout_obs.length = " + hdata.dates.length);
                                ++pdex;
                            }
                        }
                    }

                    if (NSEs.size() <= 0) {
                        return null;
                    } else {
                        NSEs.add(NSETotal / (double)NSEs.size());
                        return NSEs;
                    }
                } else {
                    logout("obsRchsList=null or obsRchsList.size() = 0");
                    logout(rch.error_log);
                    return null;
                }
            }

            public long[] _getDateList(ArrayList<Double> dateList) {
                List<Long> dates = new ArrayList();
                if (dateList.size() < 4) {
                    return null;
                } else {
                    int ys = ((Double)dateList.get(0)).intValue();
                    int ms = ((Double)dateList.get(1)).intValue() - 1;
                    int ye = ((Double)dateList.get(2)).intValue();
                    int me = ((Double)dateList.get(3)).intValue() - 1;
                    Calendar cs = Calendar.getInstance();
                    Calendar ce = Calendar.getInstance();
                    cs.set(1, ys);
                    cs.set(2, ms);
                    ce.set(1, ye);
                    ce.set(2, me);

                    while(cs.getTimeInMillis() < ce.getTimeInMillis()) {
                        dates.add(cs.getTimeInMillis());
                        cs.add(2, 1);
                    }

                    long[] dts = dates.stream().mapToLong(Long::longValue).toArray();
                    return dts;
                }
            }

            public int setSceneState(int state) {
                return state;
            }

            public List<_calswatparam> editSwatFiles(String swatLocalFolder, String pythonparamFile) {
                calibrate_param cps = new calibrate_param();
                cps.loadInfo(swatLocalFolder, paramFile);
                editswatfile esf = new editswatfile();           
                List<_calswatparam> params = esf.editparams(pythonparamFile);
                
                for(int i = 0; i < params.size(); ++i) {
                    _calswatparam p = params.get(i);
                    p.name = ((_calswatparam)cps.params.get(p.pid)).name;
                    p.suffix = ((_calswatparam)cps.params.get(p.pid)).suffix;
                    params.set(i, p);
                }

                try {      
                    esf.editSwats(swatLocalFolder, params);
                    for(_calswatparam cp : params) {
                    	logout(String.format("%f", cp.pythonvalue));
                    }
                } catch (Exception e8) {
                    logout(e8.toString());
                }

                return params;
            }

            public class HData {
                public int[] paramIndexs = null;
                public int[] reachs = null;
                public long[][] dates = null;
                public double[][] flowout_obs = null;
                public double[][] flowout_sim = null;
                public double[] paramvalues = null;
                public double[] nses = null;
                public int paramIndex = 0;

                public HData() {
                }
            }

        
    private IntWritable result = new IntWritable();

    @Override
	public void reduce(Text key, Iterable<IntWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      System.out.println("reduce func()");
      
      Configuration conf = context.getConfiguration();
      pythonUrl = conf.get("pythonurl");
      calibType = conf.get("calibtype");
      swatName = conf.get("swatname");
      calibalgo = conf.get("algorithm");
      paramFile = conf.get("paramfile");
      obsFile = conf.get("obsfile");
      System.out.println("Key is "+key);
      String strkey=key.toString();
      /*对key进行逗号分割*/
      String[] words = strkey.split(",");
      //String pythonRemoteParamFolder = ((FileSplit)context.getInputSplit()).getPath().getParent().toUri().getPath();
      //String pythonRemoteParamFile = ((FileSplit)context.getInputSplit()).getPath().getName();            
  	System.out.println("in reducer setup,set two global variables");
  	System.out.println("in reducer setup,set two global Folder: "+words[0]);
  	System.out.println("in reducer setup,set two global File: "+words[1]);
      String pythonRemoteParamFolder =words[0];
      String pythonRemoteParamFile =words[1];
      paramIndex = (new SubCellInfo())._getFileNameNoSuffix(pythonRemoteParamFile);
      pythonRemoteParamFile = pythonRemoteParamFolder + '/' + pythonRemoteParamFile;
      
      hdata.paramIndex = Integer.parseInt(paramIndex);
      logout("task_id: " + context.getTaskAttemptID());
      logout("paramIndex = " + paramIndex);
      
      //flowout-scene
      String swatRemoteFolder = String.format("%s/%s/%s/scene",
      			hdfsCalibratePath, engine, swatName);
      String swatLocalFolder = String.format("/home/hadoop/worklx/calibrate/swat-%s-%s-%s", 
      		swatName, calibalgo, paramIndex);
      
      logout("remotefolder: " + swatRemoteFolder + ";  localfolder: " + swatLocalFolder);            
      String pythonLocalParamFile = String.format("%s/param/%s.txt", swatLocalFolder, paramIndex);
      logout("pythonRemoteParamFile: " + pythonRemoteParamFile + ";  pythonLocalParamFile: " + pythonLocalParamFile);            
      String engineRemoteFolder = "/water/engine/swat";
      
      {
      	String path0 = String.format("/home/hadoop/worklx");
          createFolder(path0);
          path0 = String.format("%s/%s", path0, hdfsCalibratePath);
          createFolder(path0);
          createFolder(swatLocalFolder);
          path0 = String.format("%s/param", swatLocalFolder);
          createFolder(path0);
      }
      
      logout("start  downloading............");
      {
          ThreadSubMap t0 = new ThreadSubMap("threadSubMap", context);
          t0.start();
          downloadfolder(swatLocalFolder, swatRemoteFolder, context);
          downloadfolder(swatLocalFolder, engineRemoteFolder, context);
          downloadfile(pythonLocalParamFile, pythonRemoteParamFile, context);
          String zzEnd = String.format("%s/zzEnd", swatLocalFolder);
          String cioFil = String.format("%s/file.cio", swatLocalFolder);

          while(true)
			{
				File fs0 = new File(zzEnd);
				File fs1 = new File(cioFil);
				if(fs0.exists() && fs1.exists())
				{
					Thread.sleep(10000);
					break;
				}
				else
				{
					Thread.sleep(5000);
					progress(context);
				}
			}
			t0._Stop();
      }
      logout("end  downloading............");
      progress(context);
      
      logout("start editswat and runswat");
      List<_calswatparam> params = null;
      {	
      	logout("000000000000000000");
      	logout(paramFile);
      	logout(swatLocalFolder);
      	logout(pythonLocalParamFile);
          params = editSwatFiles(swatLocalFolder, pythonLocalParamFile);
          
          hdata.paramIndexs = new int[params.size()];
          hdata.paramvalues = new double[params.size()];
          int pdex = 0;
          logout("params.size()="+params.size());
          for(_calswatparam p : params) {
              hdata.paramIndexs[pdex] = p.getPid();
              hdata.paramvalues[pdex] = p.getPythonvalue();
              pdex++;
          }
          runSwat(swatLocalFolder, context);
      }
      logout("end editswat and runswat");
       
      List<Double> NSEs = resultDatas(swatLocalFolder, calibType, context);
      if (NSEs == null) {
          logout("aveNSE error... ");
          return;
      }
//
      progress(context);
      logout("start uploadhbase...");
      gTableName = String.format("calibrate-%s-%s", engine, swatName);
      CreateTable(gTableName, new String[]{"info"});
      uploadHBase();
      logout("end uploadhbase...");
      logout("start 整理param+nse并发送python web.................... ");
      {
          try {
          	double[][] JsonParams = new double[2][];
          	JsonParams[0] = new double[1];
          	JsonParams[0][0] = Double.parseDouble(paramIndex);
          	
          	JsonParams[1] = new double[params.size() + NSEs.size()];
              int sub = 0;
              //JsonParams[1][sub++] = Double.parseDouble(paramIndex);                    
              for(_calswatparam cp : params) {
              	JsonParams[1][sub++] = cp.pythonvalue;
              }

              for(double nse : NSEs) {
              	JsonParams[1][sub++] = nse;
              }

              String json = JSONArray.toJSONString(JsonParams);
              logout("pythonUrl= " + pythonUrl);
              logout("json= " + json);
              sendPythonResultParams(json);
          } catch (Exception e15) {
              ;
          }

          logout("end send python result params............");
          super.setup(context);
          System.gc();
      }
      
      /*reduce perform only one model run*/
      /*
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      result.set(sum);
      context.write(key, result);
      */
    }
  }
 
    public static class TokenizerMapper 
    		extends Mapper<Object, Text, Text, IntWritable> {
        private static final IntWritable one = new IntWritable(1);
        private Text word = new Text();
        public TokenizerMapper() {
        }
        @Override
		protected void setup(Mapper<Object, Text, Text, IntWritable>.Context context)
				throws IOException, InterruptedException {
			/*主要职责：获取split文件名称，传递给reduce*/
        	/*Main purpose:get name of split file and deliver it to reduce*/
    		String strDirectory=((FileSplit)context.getInputSplit()).getPath().getParent().toUri().getPath();
    		String strname=((FileSplit)context.getInputSplit()).getPath().getName();
        	System.out.println("in Map setup,combine two variables into one,send it to reducer");
        	strDirectory=strDirectory+","+strname;
    		context.write(new Text(strDirectory), new IntWritable(1)); 

		}
		public void map(Object key, Text value, Context context)
        		throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());

            while(itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                
                /*map will do nothing*/
                //context.write(word, one);
            }

        }
    }
    }
