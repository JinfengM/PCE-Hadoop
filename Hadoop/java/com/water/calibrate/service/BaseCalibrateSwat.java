package com.water.calibrate.service;


import com.water.hadoop.hadoopApi;
import com.water.hadoop.mrcalswat;
import com.water.hadoop.mrcalswat.IntSumReducer;
import com.water.hadoop.mrcalswat.TokenizerMapper;
import com.water.oper._base._filewriter;
import com.water.swat.calibrate.ObjectiveFunction;
import com.water.swat.calibrate._calswatparam;
import com.water.swat.calibrate.calibrate_param;
import com.water.swat.calibrate.editswatfile;
import java.io.BufferedInputStream;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import net.sf.json.JSONArray;
import org.apache.commons.httpclient.DefaultHttpMethodRetryHandler;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.HttpException;
import org.apache.commons.httpclient.methods.PostMethod;
import org.apache.commons.httpclient.methods.RequestEntity;
import org.apache.commons.httpclient.methods.StringRequestEntity;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.Cell;
import org.apache.hadoop.hbase.CellUtil;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.HColumnDescriptor;
import org.apache.hadoop.hbase.HTableDescriptor;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.Admin;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.client.ConnectionFactory;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.ResultScanner;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.client.Table;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.ClassUtils;
import org.springframework.web.bind.annotation.RequestParam;
import org.yaml.snakeyaml.Yaml;

@Service
public class BaseCalibrateSwat {
	//当前程序状态
	public int 	workstate = 0;
	//   hdfspath=/calibrate
	public String hdfsCalibratePath = null;
	//mirro
    public String[] calibAlgos = null;
    //0,1,2,3
    public int calibAlgoCur = 0;
    public String pythonCreateUrl_yml = null;
	public String pythonResultUrl_yml = null;
	public String pythonCreateUrl = null;
	public String pythonResultUrl = null;
	public String hdfsFolders = null;
	public String hdfsFolder = null;		
	public String engine = null;
	//  flowout, sediment
	public String calibtype = null;	
	public String swatLocalFolder = null;
	public String swatProjectName = null;
	public String jarFil = null;
	public boolean uploadScene = false;
	
	//
	public String paramFile = null;
	public String obsFile = null;
	
	//信息头
	public String[] infoHeaders = null; //获取信息头 [job, task, ....]
	
    public String getSwatProjectName() {
		return swatProjectName;
	}

	public void setSwatProjectName(String swatProjectName) {
		this.swatProjectName = swatProjectName;
	}

	public String getCalibtype() {
		return calibtype;
	}

	public void setCalibtype(String calibtype) {
		this.calibtype = calibtype;
	}

	public String getEngine() {
		return engine;
	}

	public void setEngine(String engine) {
		this.engine = engine;
	}


	/*构造函数*/
	public BaseCalibrateSwat() {
    }
    
	//test
    public int testMrJob(){
    	String testhdfsFolders = "/calibrate/swat/meijiang/calib8ktest10";
    	calibAlgoCur = 0;
    	infoHeaders = new String[]{"0"};
    	InitCalibrate_sa();
    	CreateMrJob(jarFil, 
    			engine,
    			pythonResultUrl, 
    			swatProjectName, 
    			calibtype, 
    			testhdfsFolders, 
    			calibAlgos[calibAlgoCur],
    			infoHeaders,
    			paramFile,
    			obsFile);
    	return 0;
    }
    
    public int SystemPrintInfo()
    {
    	System.out.println("hdfsCalibratePath="+hdfsCalibratePath);
        System.out.println("hdfsfolders="+hdfsFolders);
        System.out.println("hdfsfolder="+hdfsFolder);
        System.out.println("engine="+engine);
        System.out.println("calibtype="+calibtype);
        System.out.println("pythoncreateurl="+pythonCreateUrl);
        System.out.println("pythonresulturl="+pythonResultUrl);
        System.out.println("swatlocalfolder="+swatLocalFolder);
        System.out.println("swatProjectName="+swatProjectName);
        System.out.println("jarFil="+jarFil);
        System.out.println("calibalgos="+calibAlgos);
        System.out.println("calibAlgoCur="+calibAlgoCur);
        return 1;
    }
    
    public int _GetNextAlgorithmIndex()
    {
    	calibAlgoCur++;
    	if (calibAlgoCur >= calibAlgos.length || calibAlgos[calibAlgoCur].isEmpty()) 
    	{
    		calibAlgoCur = 0;
    		return 0;
    	}
    	return calibAlgoCur;
    }
    
    public String _getYml(Map map, String key, String defaultStr)
    {
    	try{
    		return map.get(key).toString();
    	}catch(Exception e)
    	{
    		return defaultStr;
    	}
    }
    
    public Configuration _getConfiguration()
    {
    	Configuration cfg = new Configuration(true);
//        cfg.set("fs.hdfs.impl.disable.cache", "true");
        
        cfg.set("mapreduce.app-submission.cross-platform", "true");
        return cfg;
    }
    

    public int InitCalibrate(String ymlfile)
    {
	    try{
	    	Yaml yml = new Yaml();    
			
			String ymlfile0 = ClassUtils.getDefaultClassLoader().getResource("").getPath() + ymlfile;
			System.out.println("application.xml = " + ymlfile0);
			InputStream reader = new FileInputStream(new File(ymlfile0));
			//yml读取配置文件,指定返回类型为Map,Map中value类型为LinkedHashMap
			Map map = yml.loadAs(reader, Map.class);
			Map mapCalib = (Map)map.get("calibrate");
	    	
			hdfsCalibratePath = _getYml(mapCalib, "hdfscalibratepath", "/calibrate");
			engine = _getYml(mapCalib, "engine", "swat");
			calibtype = _getYml(mapCalib, "calibtype", "flowout");
			hdfsFolders = _getYml(mapCalib, "hdfdfolders", "multihdfs");
			hdfsFolder = _getYml(mapCalib, "hdfsFolder", "singlehdfs");
			paramFile = _getYml(mapCalib, "paramfile", "sa_flowout.txt");
			obsFile = _getYml(mapCalib, "obsfile", "flowout.obs");
			pythonCreateUrl_yml = _getYml(mapCalib, "pythoncreateurl", ""); 
			pythonResultUrl_yml = _getYml(mapCalib, "pythonresulturl", "");	
			
			pythonCreateUrl =  pythonCreateUrl_yml + calibAlgoCur;
			pythonResultUrl =  pythonResultUrl_yml + calibAlgoCur;
        			
			swatLocalFolder = _getYml(mapCalib, "swatlocalfolder", "");
			swatProjectName = _getYml(mapCalib, "swatprojectname", "");
			jarFil = _getYml(mapCalib, "jarfil", "");
			uploadScene = Boolean.parseBoolean(_getYml(mapCalib, "uploadscene", "false"));
		
			String algoArrs = _getYml(mapCalib, "algorithms", "morris");
			calibAlgos = algoArrs.trim().split(",");
			
	        if (calibAlgoCur >= calibAlgos.length || calibAlgos[calibAlgoCur].isEmpty()) 
	            return 0;
			
	        SystemPrintInfo();
	        
	    } catch (Exception var2) {
	        ;
	    }

	    return 1;
    }
    
    public int InitCalibrate_sa() {
        return InitCalibrate("application-sa.yml");
    }
      
    public int InitCalibrate_bo() {
    	return InitCalibrate("application-bo.yml");
    }
    
    public String _getInitInfo()
    {
    	String cl = " </br>";
    	String info = "";
    	
    	info = info + "//正式启动敏感性分析-配置application.yml即可 " + cl;
    	info = info + "url:/calibrate/calibrate-sencreate" + cl;
    	
    	info = info + "//其中一个算法敏感性分析完毕，python返回数据  " + cl;
    	info = info + "url:/calibrate/calibrate-end  " + cl;
    	
    	info = info + "//集群保存10个参数txt，进行集群测试。返回python result 10个post  " + cl;
    	info = info + "url:/calibrate/calibrate-test10  " + cl;
    	
    	info = info + "//集群路径-率定目录-默认" + cl;    			
    	info = info + "hdfsCalibratePath="+hdfsCalibratePath + cl;
    	
    	info = info + "//单job多task hdfs文件路径, --默认即可" + cl;
    	info = info + "hdfsfolders="+ hdfsFolders + cl;
    	info = info + "//单job多task hdfs文件路径, --默认即可" + cl;
    	info = info + "hdfsfolder="+ hdfsFolder + cl;
    	
    	info = info + "//paramfile, --" + cl;
    	info = info + "paramfile="+ paramFile + cl;
    	
    	info = info + "//obsfile" + cl;
    	info = info + "obsfile="+ obsFile + cl;
    	
    	info = info + "//引擎" + cl;
    	info = info + "engine="+engine + cl;
    	
    	info = info + "//率定内容 flowout, sediment, waterquality" + cl;
    	info = info + "calibtype="+calibtype + cl;
    	
    	info = info + "//启动网址 pythoncreateurl+n," + cl;
    	info = info + "pythoncreateurl="+pythonCreateUrl + cl;
    	
    	info = info + "//返回网址 pythonresulturl+0,1,2" + cl;
    	info = info + "pythonresulturl="+pythonResultUrl + cl;
    	
    	info = info + "//场景目录，里边需要包含param文件夹(flowout.obs实测值, flowout.txt 参数 详见样本文件)" + cl;
    	info = info + "swatlocalfolder="+swatLocalFolder + cl;
    	
    	info = info + "//项目名称(关系到集群运算等)" + cl;
    	info = info + "swatProjectName="+swatProjectName + cl;
    	
    	info = info + "//集群mapred运算包" + cl;
    	info = info + "jarFil="+jarFil + cl;
    	
    	info = info + "//算法总数" + cl;
    	info = info + "calibalgos="+calibAlgos + cl;
    	
    	info = info + "calibAlgoCur="+calibAlgoCur + cl;
        return info;
    }
    public int CreateCalibrate() {
        try {
        	
            Configuration cfg = _getConfiguration();
            System.out.println("begin upload swat folder , it maybe take 10~30 mins, please waiting...");
            if (uploadScene && swatLocalFolder != null && !swatLocalFolder.isEmpty()) {
            	String remoteFolder = String.format("%s/%s/%s/scene", 
            			hdfsCalibratePath, engine, swatProjectName);
            	FileSystem hdfs = FileSystem.get(cfg);
    	        Path path = new Path(remoteFolder);
    	        boolean bflag = hdfs.exists(path);
    	        if (bflag) {
    	            hdfs.delete(path, true);
    	        }
    	        hdfs.mkdirs(path);
    	        Path pathParam = new Path(remoteFolder+"/param");
    	        hdfs.mkdirs(pathParam);
                hadoopApi.uploadfolder_static(remoteFolder, swatLocalFolder, cfg);
            }
            
            System.out.println("end upload swat folder ,  please check hdfs files...");

            calibrate_param cp = new calibrate_param();
            String paramJson = cp.loadInfo_Json(swatLocalFolder, paramFile);
            System.out.println(paramJson);
            String pythonJson = sendPythonCreateCalibrate(paramJson);
            if (pythonJson == null) {
                return -2;
            } else {
            	OperatePythonMsg(pythonJson);
                return 1;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return -1;
        }
    }

    public int EndCalibrateSen(String pythonJson) {
        try {
            String tableName = String.format("calibrate-%s-%s", engine, swatProjectName);
            Connection hBaseConn = ConnectionFactory.createConnection(HBaseConfiguration.create());
            Admin hAdmin = hBaseConn.getAdmin();
            TableName tbName = TableName.valueOf(tableName);
            String[] cols = new String[]{"info"};
            if (hAdmin.tableExists(tbName)) {
                System.out.println("table is exists!");
            } else {
            	HTableDescriptor hTableDescriptor = new HTableDescriptor(tbName);  
                
                for(String col:cols){  
                    HColumnDescriptor hColumnDescriptor = new HColumnDescriptor(col);  
                    hTableDescriptor.addFamily(hColumnDescriptor);  
                }  
                hAdmin.createTable(hTableDescriptor);
            }

            Table table = hBaseConn.getTable(TableName.valueOf(tableName));
            String strKey = String.format("%s-param-%s", calibtype, calibAlgos[calibAlgoCur]);
            
            List<Put> putList = new ArrayList();
            Put p = new Put(Bytes.toBytes(strKey));
         
            p.addColumn(Bytes.toBytes("info"), 
            		Bytes.toBytes("pythonresult"), 
            		Bytes.toBytes(pythonJson));
            
            p.addColumn(Bytes.toBytes("info"), 
            		Bytes.toBytes("algorithm"), 
            		Bytes.toBytes(calibAlgos[calibAlgoCur]));
            
            putList.add(p);
            table.put(putList);
            table.close();
            hBaseConn.close();
        } catch (Exception var13) {
            ;
        }

        return 1;
    }

    public int OperatePythonMsg(String pythonJson) {
        Object var2 = null;
        try {
            double[][] datas = splitPythonJson(pythonJson);
            if (datas.length < 2) 
            	return -1;
            
            ContinueCalibrateSwat(datas);
            return 1;
        } catch (Exception var4) {
            var4.printStackTrace();
            return -1;
        }
    }

    public int ContinueCalibrateSwat(double[][] datas) {
        try {
        	//需要信息头
        	if(infoHeaders==null)
        		return 0;
            Configuration cfg = _getConfiguration();
            calibrate_param cp = new calibrate_param();
            cp.loadInfo(swatLocalFolder, paramFile);
            
            String remoteHdfsFolder = null;
            
        	System.out.println("single job and mutilt task, one folder and some files start  proc and upload hdfs ....");
        	remoteHdfsFolder =CreateRemoteHdfs_SingleJobMultiTask(cfg, datas, cp.params);
            System.out.println("a hdfs folder and some files end  proc and upload hdfs ....");
            
            CreateMrJob(jarFil, 
        			engine,
        			pythonResultUrl, 
        			swatProjectName, 
        			calibtype, 
        			remoteHdfsFolder,
        			calibAlgos[calibAlgoCur],
        			infoHeaders,
        			paramFile,
        			obsFile);
            
            return 1;
        } catch (Exception var4) {
            var4.printStackTrace();
            return -1;
        }
    }

    public int EndCalibrateSwat(double[][] datas) {
        int i, j;
        ArrayList pythonParams = new ArrayList();

        try {
            for(j = 0; j < datas.length; ++j) {
                _calswatparam cp = new _calswatparam();
                cp.pid = (int)datas[0][j];
                cp.pythonvalue = datas[1][j];
                pythonParams.add(cp);
            }

            calibrate_param cps = new calibrate_param();
            cps.loadInfo(swatLocalFolder, paramFile);

            for(i = 0; i < pythonParams.size(); ++i) {
                _calswatparam p = (_calswatparam)pythonParams.get(i);
                p.name = ((_calswatparam)cps.params.get(p.pid)).name;
                p.suffix = ((_calswatparam)cps.params.get(p.pid)).suffix;
                pythonParams.set(i, p);
            }

            editswatfile esf = new editswatfile();
            esf.editSwats(swatLocalFolder, pythonParams);
            System.out.println("calibrate over.!!!");
            return 1;
        } catch (Exception e) {
            e.printStackTrace();
            return -1;
        }
    }

    public String sendPythonCreateCalibrate(String paramJson) {
        System.out.println("send params ing:==============================");
        System.out.println("url: " + pythonCreateUrl);
        System.out.println(paramJson);
        String url = pythonCreateUrl;
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
            System.out.println("python statuscode= " + statusCode);
            if (statusCode == 200) {
                byte[] responseBody = postMethod.getResponseBody();
                resultMsg = new String(responseBody);

                try {
                    _filewriter fw = new _filewriter("d:/jsonpython.txt");
                    fw.append(resultMsg + "\r\n");
                    fw.flush();
                    fw.close();
                } catch (Exception e) {
                    e.getStackTrace();
                }

                System.out.println("result:==========get python param datas and analysis====================");
                return resultMsg;
            }
        } catch (HttpException e1) {
            System.out.println("Please check your provided http address!");
            e1.printStackTrace();
            return resultMsg;
        } catch (IOException e2) {
            return resultMsg;
        } finally {
            postMethod.releaseConnection();
        }

        return null;
    }
    public String getRemoteHdfsFolder(String algorithmName, String[] infoHeaders)
    {
    	if(infoHeaders==null)
    		return null;
    	
    	String remoteHdfsFolder = null;
    	
    	
		remoteHdfsFolder = String.format("%s/%s/%s/%s-%s-%s", 
        		hdfsCalibratePath, engine, swatProjectName, 
        		hdfsFolders, algorithmName, infoHeaders[0]);    	
    	return remoteHdfsFolder;
    }
    
    public String getRemoteOutputFolder(String algorithmName, String[] infoHeaders)
    {
    	if(infoHeaders==null)
    		return null;
    	
    	String remoteOutputFolder = null;
		remoteOutputFolder = String.format("%s/%s/%s/output/output-%s-%s", 
        		hdfsCalibratePath, engine, swatProjectName, 
        		algorithmName, infoHeaders[0]);
    	
    	return remoteOutputFolder;
    }
    
    public String CreateRemoteHdfs_SingleJobMultiTask(Configuration cfg, double[][] datas, List<_calswatparam> params)
    {
    	try {
    		int i, j;
			if(datas==null || infoHeaders==null)
				return null;
			int fileIndexStart = Integer.parseInt(infoHeaders[0]);
			if(fileIndexStart<0)
				fileIndexStart = 0;
			//fileStart
			fileIndexStart = fileIndexStart*(datas.length-1);
			//calibindex_yml
			List<_calswatparam> pythonParams = new ArrayList<_calswatparam>();			
			//param头标
			for(i=0; i<datas[0].length; i++)
			{
				int index = (int)datas[0][i];
				_calswatparam cwp = new _calswatparam();
				cwp.pid = params.get(index).pid;
				pythonParams.add(cwp);
			}

			FileSystem hdfs = FileSystem.get(cfg);
			String remoteHdfsFolder = getRemoteHdfsFolder(calibAlgos[calibAlgoCur], infoHeaders);
			
	        Path path = new Path(remoteHdfsFolder);
	        boolean bflag = hdfs.exists(path);
	        //
	        if (bflag) {
	            hdfs.delete(path, true);
	        }
	        hdfs.mkdirs(path);
	        
	        //创建小文件
	        String content="", title="";
	        BufferedInputStream in =null;
	        Path outputPath = null;
	        FSDataOutputStream out =null;
	        	
	        //pid
	        for(j=0; j<pythonParams.size(); j++)
        	{
	        	if(j>0)
	        		title+=',';
        		title += pythonParams.get(j).pid;
        	}
	        title += "\r\n";
	        
	        //data
	        
	        System.out.println("集群生成文件中...");
	        System.out.println("datas.length = " + datas.length);
	        //datas.length
	        for (i=1; i<datas.length; i++) 
	        {
	        	content = title;//头标
	        	for(j=0; j<pythonParams.size(); j++)
	        	{
	        		if(j>0)
		        		content +=',';
		        	content += String.format("%.4f", datas[i][j]);
	        	}	
	        	System.out.println(content);
	            in=new BufferedInputStream(new ByteArrayInputStream(content.getBytes()));
	            outputPath=new Path(String.format("%s/%d.txt", remoteHdfsFolder, i-1+fileIndexStart));
	            out=hdfs.create(outputPath);
	            IOUtils.copyBytes(in,out,4096,true);	            
	            //
                if((i-1)%1000==0)
                	System.out.println(String.format("%s/%d.txt", remoteHdfsFolder, i-1+fileIndexStart));
	        }

            System.out.println(datas.length - 1 + "文件配置完成");
            return remoteHdfsFolder;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    public String CreateRemoteHdfs_MultiJobSingleTask(Configuration cfg, double[][] datas, List<_calswatparam> params)
    {
    	try {
    		int i, j;
			if(datas==null || infoHeaders==null || infoHeaders.length<1)
				return null;
			//calibindex_yml
			List<_calswatparam> pythonParams = new ArrayList<_calswatparam>();			
			//param头标
			for(i=0; i<datas[0].length; i++)
			{
				int index = (int)datas[0][i];
				_calswatparam cwp = new _calswatparam();
				cwp.pid = params.get(index).pid;
				pythonParams.add(cwp);
			}

			FileSystem hdfs = FileSystem.get(cfg);
			String remoteHdfsFolder = getRemoteHdfsFolder(calibAlgos[calibAlgoCur], infoHeaders);
					
			try{
				Path path = new Path(remoteHdfsFolder);
		        boolean bflag = hdfs.exists(path);
		        if (bflag) {
		            hdfs.delete(path, true);
		        }
		        hdfs.mkdirs(path);
			}catch(Exception e){
				System.out.println("delete remode folder error=====================================================");
			}
	        	       
	        //创建小文件
	        String content="", title="";
	        BufferedInputStream in =null;
	        Path outputPath = null;
	        FSDataOutputStream out =null;
	        	
	        //pid
	        for(j=0; j<pythonParams.size(); j++)
        	{
	        	if(j>0)
	        		title+=',';
        		title += pythonParams.get(j).pid;
        	}
	        title += "\r\n";
	        
	        //data
	        System.out.println("集群生成文件中...");
	        System.out.println("datas.length = " + datas.length);
	        //datas.length
	        for (i=1; i<datas.length; i++) 
	        {
	        	content = title;//头标
	        	for(j=0; j<pythonParams.size(); j++)
	        	{
	        		if(j>0)
		        		content +=',';
		        	content += String.format("%.4f", datas[i][j]);
	        	}	            
	            in=new BufferedInputStream(new ByteArrayInputStream(content.getBytes()));
	            outputPath=new Path(String.format("%s/%s.txt", remoteHdfsFolder, infoHeaders[0]));
	            
	            out=hdfs.create(outputPath);
	            IOUtils.copyBytes(in,out,4096,true);
	        }

            System.out.println(datas.length - 1 + "文件配置完成");
            return remoteHdfsFolder;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    public int CreateMrJob(String jarFil,	//jar包 
    						String engine,	//引擎
    						String pythonResultUrl,	//返回url 
    						String swatname, 	//swat场景名称
    						String calibtype, 	//类型 flowout, sediment, wq
    						String jobRemoteFolder, 	//启动job文件
    						String algoName,
    						String[] infoHeaders,
    						String paramFile,
    						String obsFile)	//算法名称 
    {
    	long starttime = System.currentTimeMillis();
        try {
            Configuration cfg = _getConfiguration();
            
            cfg.set("engine", engine);
            cfg.set("pythonurl", pythonResultUrl);
            cfg.set("calibtype", calibtype);
            cfg.set("swatname", swatname);
            cfg.set("algorithm", algoName);
            cfg.set("paramfile", paramFile);
            cfg.set("obsfile", obsFile);
                                    
            String jobName = String.format("calibrate-swat-%s", swatname);
            if(infoHeaders!=null){
            	jobName = String.format("calibrate-swat-%s-%s", swatname, infoHeaders[0]);
            }
            
            Job job = Job.getInstance(cfg);
            job.setJobName(jobName);
            /* MJF:use reduce to run model */
            //job.setNumReduceTasks(100);
            /* MJF:only use map to run model */
            job.setNumReduceTasks(0);
            job.setJarByClass(mrcalswat.class);
            job.setMapperClass(TokenizerMapper.class);
            /*setReducerClass和setCombinerClass不能共存*/
            //job.setCombinerClass(IntSumReducer.class);
            job.setReducerClass(IntSumReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(IntWritable.class);
            job.setJar(jarFil);
            String remoteFolder =jobRemoteFolder;            
            String outputFolder = getRemoteOutputFolder(algoName, infoHeaders);
            Path outputfolder = new Path(outputFolder);
            if (outputfolder.getFileSystem(cfg).exists(outputfolder)) {
                outputfolder.getFileSystem(cfg).delete(outputfolder, true);
            }
            FileInputFormat.addInputPath(job, new Path(remoteFolder));
            FileOutputFormat.setOutputPath(job, new Path(outputFolder));
            job.waitForCompletion(true);
            long endtime = System.currentTimeMillis();
            long costtime=endtime-starttime;
            System.out.println(costtime);
            /*写入计算时间*/
            String strcontent="starttime,"+starttime +",endtime,"+endtime+",costtime,"+costtime+"\r\n";
             writecost("costtime.txt",strcontent );                        
            return 1;
        } catch (Exception e) {
            e.printStackTrace();
            return -1;
        }
    }
    public  void writecost(String file, String conent) {
    	BufferedWriter out = null;
    	try {
    	out = new BufferedWriter(new OutputStreamWriter(
    	new FileOutputStream(file, true)));
    	out.write(conent+"\r\n");
    	} catch (Exception e) {
    	e.printStackTrace();
    	} finally {
    	try {
    	out.close();
    	} catch (IOException e) {
    	e.printStackTrace();
    	}
    	}
    	}
    public String simulatePythonJson() {
        int[] pids = new int[]{0, 1, 5, 10, 22};
        List<_calswatparam> params = new ArrayList<_calswatparam>();

        int i;
        for(i = 0; i < pids.length; ++i) {
            _calswatparam cp0 = new _calswatparam();
            cp0.pid = pids[i];
            params.add(cp0);
        }

        String resultJsons = "[[";

        int j;
        for(j = 0; j < params.size(); ++j) {
            if (j > 0) {
                resultJsons = resultJsons + ',';
            }

            resultJsons = resultJsons + ((_calswatparam)params.get(j)).pid;
        }

        resultJsons = resultJsons + "]";

        for(i = 0; i < 10; ++i) {
            resultJsons = resultJsons + ",[";

            for(j = 0; j < params.size(); ++j) {
                if (j > 0) {
                    resultJsons = resultJsons + ',';
                }

                resultJsons = resultJsons + String.format("%.4f", Math.random() * 10.0D);
            }

            resultJsons = resultJsons + ']';
        }

        resultJsons = resultJsons + ']';
        return resultJsons;
    }
    public double[][] splitPythonJson(String pythonJson) 
    {
        int i, j, dataCount=0, paramCount=0, headCount=0;
        try {
            JSONArray pythonArrs = JSONArray.fromObject(pythonJson);
            dataCount = pythonArrs.size();
            if (dataCount < 3)	//信息头、参数索引、值、值、值..... 
                return null;
        	//获取信息头 [job, task, ....]
        	JSONArray headerArrs = JSONArray.fromObject(pythonArrs.getString(0));
        	headCount = headerArrs.size();
        	infoHeaders = new String[headCount];
        	for(i = 0; i < headCount; ++i) {
        		infoHeaders[i] = headerArrs.getString(i);
            }
        	
        	//获取参数索引
            JSONArray pidArrs = JSONArray.fromObject(pythonArrs.getString(1));
            paramCount = pidArrs.size();
            if (paramCount < 1) 
                return null;
            
            double[][] datas = new double[dataCount-1][paramCount];
            
            //参数索引
            for(j = 0; j < paramCount; ++j) {
                datas[0][j] = Double.parseDouble(pidArrs.getString(j).trim());
            }
            //值、值。。。
            for(i = 2; i < dataCount; ++i) {
                JSONArray dArrs = JSONArray.fromObject(pythonArrs.getString(i));
                int count = dArrs.size();
                if (count == paramCount) {
                    for(j = 0; j < paramCount; ++j) {
                        datas[i-1][j] = Double.parseDouble(dArrs.getString(j).trim());
                    }
                }
            }

            return datas;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public String simulatePythonJson_End() {
        String resultJson = "[[0,1,5,10,22],[8.1,2.1,2.1,7.1,5.1]]";
        return resultJson;
    }    
    
    public String wGetResultData(String operate,
					    		String engine,
					    		String projectname,
					    		String calibtype,
					    		String algo) 
    {
    	String JsonTotal = "";
        String tableName = String.format("%s-%s-%s", operate, engine, projectname);
        ResultScanner resultScanner = null;
        try {
            long starttime = System.currentTimeMillis();
            Connection hBaseConn = ConnectionFactory.createConnection(HBaseConfiguration.create());
            Table table = hBaseConn.getTable(TableName.valueOf(tableName));
            
            String StartKey = String.format("%s-sim-%s-00000", calibtype, algo);
            String EndKey = String.format("%s-sim-%s-99999", calibtype, algo);

            Scan scan = new Scan();
            scan.setStartRow(Bytes.toBytes(StartKey));
            scan.setStopRow(Bytes.toBytes(EndKey));
            resultScanner = table.getScanner(scan);
            String JsonCell="";           
			for (Result res : resultScanner) 
			{		
            	String row = new String(res.getRow());
            	JsonCell = "";
            	for (Cell ce : res.rawCells()) 
            	{
                    String rowkey = new String(CellUtil.cloneRow(ce));
                    String family = new String(CellUtil.cloneFamily(ce));
                    String qualifier = new String(CellUtil.cloneQualifier(ce));
                    String value = new String(CellUtil.cloneValue(ce));
                    String line = String.format("key=%s, column=%s:%s value=%s", rowkey, family, qualifier, value);
                    System.out.println(line);
                    String colValue = String.format("\"%s\": %s,", qualifier, value);
                    JsonCell = JsonCell + colValue;
                }
                JsonCell = String.format("{%s},", JsonCell.substring(0, JsonCell.length() - 1));
                JsonTotal = JsonTotal + JsonCell;
                System.out.println(JsonTotal.length());
            }
			if(!JsonTotal.isEmpty())
				JsonTotal = '[' + JsonTotal.substring(0, JsonTotal.length() - 1) + ']';
            
          //file 得到文件夹目录
			File file0 = new File(swatLocalFolder);
	        String strParentDirectory = file0.getParent();
	        System.out.println(strParentDirectory);
	        String algoInfoFile = String.format("%s/%s.txt", strParentDirectory, algo);
            
            _filewriter fw = new _filewriter(algoInfoFile);
            fw.append(JsonTotal + "\r\n");
            fw.flush();
            fw.close();
            long endtime = System.currentTimeMillis();
            long val = endtime - starttime;
            System.out.println(val);
            table.close();
        } catch (IOException var23) {
            var23.printStackTrace();
        }

        return JsonTotal;
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

    public static void main(String[] args) {
    	CalibrateSwat cs = new CalibrateSwat();
//        cs.InitCalibrate_bo();
//        String jsonParam = "[[3], [0,1,2,3,4,5,6,7],[1.2,2.1,2.1,2.1,2.1,2.1,2.1,2.1]]";
//        cs.OperatePythonMsg(jsonParam);
//    	cs.testMrJob();
//        cs.CreateCalibrate();
//        cs.testMrJob();//                
//        cs.wGetResultData("calibrate", "swat", "meijiang", "flowout", "morris");	//
    	
        
    }
}
