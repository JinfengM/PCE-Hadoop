package com.water.pub;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapred.*;

public class HadoopJob {

	 public JobStatus[] getJobs()
	    {
	        JobStatus[] jobStatusArrs = null;
	        try {
	            Configuration conf = new Configuration(true);
	            JobClient jobClient = new JobClient(null, conf);//初始化jobclient

	            jobStatusArrs = jobClient.getAllJobs();
	            int size = jobStatusArrs.length;
	            
	            for(int i=340; i<370; i++)
	            {
	            	JobID jobID = JobID.forName("job_1591953610073_0"+i);
		            RunningJob job = jobClient.getJob(jobID);
		            if(job!=null)
		            {
		                job.killJob();
		            }
	            }
	            
	            
	            
//	            int i, j;
//	            int size = jobStatusArrs.length;
//	            String[] jobStateArrs = {"running", "succeeded", "failed", "prep", "killed","else"};
//	            for(i=0; i<size; i++)
//	            {
//	                JobStatus jobStatus = jobStatusArrs[i];
//	                int jobState = jobStatus.getRunState();
//	                System.out.println("job Id:"+jobStatus.getJobID());
//	                System.out.println(String.format("job State = %s", jobStateArrs[jobState]));
//	                System.out.println("job Name:"+jobStatus.getJobName());
//	                System.out.println("job User:"+jobStatus.getUsername());
//	                System.out.println("job Start-time:"+jobStatus.getStartTime());
//	                System.out.println("job End-time:"+jobStatus.getFinishTime());
//	                System.out.println("job Url:"+jobStatus.getTrackingUrl());
//	                System.out.println();
	//
//	                //Mapper
//	                TaskReport[] mtp = jobClient.getMapTaskReports(jobStatus.getJobID());
//	                for(j=0;j<mtp.length;j++)
//	                {
//	                    System.out.println("task id:"+mtp[j].getTaskId());
//	                    System.out.println("task Start-time:"+mtp[j].getStartTime());
//	                    System.out.println("task Finish-time:"+mtp[j].getFinishTime());
//	                    System.out.println("task Progress:"+mtp[j].getProgress());
//	                    System.out.println("task Status:"+mtp[j].getCurrentStatus());
//	                    System.out.println("task Counters:"+mtp[j].getCounters());
//	                }
//	                System.out.println("Reduce task:");//reduce详情信息
//	                TaskReport[] rtp = jobClient.getReduceTaskReports(jobStatus.getJobID());
//	                for(j=0;j<rtp.length;j++) {
//	                    System.out.println("task id:" + rtp[j].getTaskId());
//	                    System.out.println("task Start-time:" + rtp[j].getStartTime());
//	                    System.out.println("task Finish-time:" + rtp[j].getFinishTime());
//	                    System.out.println("task Progress:" + rtp[j].getProgress());
//	                    System.out.println("task Status:" + rtp[j].getCurrentStatus());
//	                    System.out.println("task Counters:" + rtp[j].getCounters());
//	                }
//	            }

	        }catch (Exception e) {
	            e.printStackTrace();
	        }
	        return jobStatusArrs;
	    }

	    public int killJob(String jobStr)
	    {
	        try
	        {
	            Configuration conf = new Configuration(true);
	            JobClient jobClient = new JobClient(null, conf);//初始化jobclient
	            JobID jobID = JobID.forName(jobStr);
	            RunningJob job = jobClient.getJob(jobID);
	            if(job!=null)
	            {
	                job.killJob();
	                return 1;
	            }
	        }catch(Exception e)
	        {
	            e.printStackTrace();
	            return -1;
	        }
	        return 0;
	    }
	    public static void main(String[] args) throws Exception {
	        try {
	            new HadoopJob().getJobs();

	        }catch (Exception e){
	            e.printStackTrace();
	        }
	    }

}
