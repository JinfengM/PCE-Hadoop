package com.water.hadoop;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import java.io.IOException;
import org.apache.commons.httpclient.DefaultHttpMethodRetryHandler;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.HttpException;
import org.apache.commons.httpclient.methods.PostMethod;
import org.apache.commons.httpclient.methods.RequestEntity;
import org.apache.commons.httpclient.methods.StringRequestEntity;

public class testpost {
    public testpost() {
    }

    public static void main(String[] args) throws Exception {
        System.out.println("System started");
        String url = "http://172.16.254.80:9999/BOPARAM0";
//        String url = "http://localhost:8080/bocontinue";
//        String url = "http://localhost:9999/LUSEINIT";
        
        HttpClient httpClient = new HttpClient();
        PostMethod postMethod = new PostMethod(url);
        JSONArray jsonArray = new JSONArray();

        for(int i = 0; i < 3; ++i) {
            JSONObject jsonObject = new JSONObject();
            jsonObject.put("name", "name" + 2 * i);
            jsonObject.put("min", 18 + i);
            jsonObject.put("max", 20 + i);
            jsonArray.add(jsonObject);
        }

        System.out.println("System started");
        String paramJson = jsonArray.toString();
//        String paramJson = "[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3]";
        System.out.println(paramJson);
        try {
            postMethod.getParams().setParameter("http.method.retry-handler", new DefaultHttpMethodRetryHandler());
            postMethod.getParams().setContentCharset("UTF-8");
            RequestEntity se = new StringRequestEntity(paramJson, "application/json", "UTF-8");
            postMethod.setRequestEntity(se);
            int ctimeout = httpClient.getHttpConnectionManager().getParams().getConnectionTimeout();
            int stimeout = httpClient.getHttpConnectionManager().getParams().getSoTimeout();
            httpClient.getHttpConnectionManager().getParams().setConnectionTimeout(60000);
            httpClient.getHttpConnectionManager().getParams().setSoTimeout(60000);
            System.out.println("System executeMethod");
            int statusCode = httpClient.executeMethod(postMethod);
            System.out.println("result:==============================" + statusCode);
            if (statusCode == 200) {
                byte[] responseBody = postMethod.getResponseBody();
                String resultMsg = new String(responseBody);
                System.out.println("result:==============================");
                System.out.println(resultMsg);
                return;
            }

            System.out.println("failed:==============================");
        } catch (HttpException e) {
            System.out.println("Please check your provided http address!");
            e.printStackTrace();
            return;
        } catch (IOException e) {
            return;
        } finally {
            postMethod.releaseConnection();
        }

    }
}
