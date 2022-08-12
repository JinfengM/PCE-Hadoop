package com.water.pub;

import java.io.IOException;
import org.apache.commons.httpclient.DefaultHttpMethodRetryHandler;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.HttpException;
import org.apache.commons.httpclient.methods.PostMethod;


public class api {
    public api() {
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
        } catch (IOException e2) {
            ;
        } finally {
            postMethod.releaseConnection();
        }

        return (String)resultMsg;
    }

    public static void main(String[] args) {
    }
}