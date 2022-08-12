//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package com.water.calibrate.controller;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.net.URLDecoder;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.util.ClassUtils;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.context.request.RequestAttributes;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;
import org.springframework.web.servlet.ModelAndView;

import com.alibaba.fastjson.JSONArray;
import com.water.calibrate.luses.luse;
import com.water.calibrate.service.CalibrateSwat;
import com.water.oper._base._api;

@RestController
@EnableAutoConfiguration
@SpringBootApplication(
    scanBasePackages = {"com.water"}
)
public class CalibrateSwatServer {
    @Autowired
    private CalibrateSwat calibSwat;
    
    @Autowired
    private HttpServletRequest pyRequest;

    public CalibrateSwatServer() {
    }

    public static void main(String[] args) {
        SpringApplication.run(CalibrateSwatServer.class, args);
    }
    
//    @ResponseBody
    @RequestMapping("/")
    public ModelAndView homepage() {
//    	System.out.println("homepage==============================");
//    	String info = calibSwat._getInitInfo();
    	return new ModelAndView("calibrate.html");       
    }

    @ResponseBody
    @RequestMapping({"/calibrate-test10"})
    public String calibrateTest10() {
    	calibSwat.calibAlgoCur = 0;
        int error = calibSwat.InitCalibrate_sa();
        if (error == 0) {
            return "failed-calibrate=-1";
        }

        calibSwat.testMrJob();
        return "calibrate-test";
    }

    @RequestMapping({"/sencreate_sa"})
    public String calibrateStart_sa() {
        System.out.println("calibrate:sencreate_sa");
        try {
        	if(calibSwat.workstate!=0)
        		return "calibrate working...";
            calibSwat.calibAlgoCur = 0;
            int error = calibSwat.InitCalibrate_sa();
            if (error == 0) {
            	calibSwat.workstate=0;
                return "failed-calibrate=-1";
            }

            calibSwat.CreateCalibrate();
            calibSwat.workstate=0;
        } catch (Exception e) {
            ;
        }

        return "ok";
    }
    
    @RequestMapping({"/sencreate_bo"})
    public String calibrateStart_bo() {
        System.out.println("calibrate:sencreate_bo");
        try {
        	if(calibSwat.workstate!=0)
        		return "calibrate working...";
            calibSwat.calibAlgoCur = 0;
            int error = calibSwat.InitCalibrate_bo();
            if (error == 0) {
            	calibSwat.workstate=0;
                return "failed-calibrate=-1";
            }
            calibSwat.CreateCalibrate();
            calibSwat.workstate=0;
        } catch (Exception e) {
            ;
        }

        return "ok";
    }

    @ResponseBody
    @RequestMapping(
        value = {"/senend"},
        method = {RequestMethod.POST},
        produces = {"application/json;charset=UTF-8"}
    )
    
    public String calibrateAlgoEnd(@RequestBody String jsonParam) throws IOException {
        System.out.println("calibrate-senend");
        calibSwat.EndCalibrateSen(jsonParam);
        int cur = calibSwat._GetNextAlgorithmIndex();
        if (cur == 0) {
            return "calibrate-senend, next algo not create";
        } else {
            calibSwat.CreateCalibrate();
            return "calibrate-senend, , next algo created";
        }
    }
    
    @RequestMapping({"/resultdata"})
    public String resultdata(@RequestParam(name = "operate") String operate,
    		@RequestParam(name = "engine") String engine,
    		@RequestParam(name = "projectname") String projectname,
    		@RequestParam(name = "calibtype") String calibtype,
    		@RequestParam(name = "algo") String algo) 
    {	
        System.out.println("for resultdata... ");
        String json = "";
        try {
        	json = calibSwat.wGetResultData(operate, engine, projectname, calibtype, algo);	//
        } catch (Exception e) {
            ;
        }

        return json;
    }
    
    @RequestMapping({"/initinfo"})
    public String initinfo() {
        try {
        	System.out.println("initinfo=====================");
            int error = calibSwat.InitCalibrate_sa();
        } catch (Exception e) {
            ;
        }

        return JSONArray.toJSONString(calibSwat);
    }
    
    @ResponseBody
    @RequestMapping(
        value = {"/bocontinue"},
        method = {RequestMethod.POST},
        produces = {"application/json;charset=UTF-8"}
    )    
    public void boContinue(@RequestBody String jsonParam) throws IOException {
        System.out.println("bocontinue");
        try {
        	CalibrateSwat cSwat = new CalibrateSwat();
        	cSwat.InitCalibrate_bo();
        	//return msg before,  for python no waitting
        	/*接收到新的56组参数集后，先提前返回给Tornado，避免python等待*/
        	ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
            HttpServletRequest request = attributes.getRequest();
            HttpServletResponse response = attributes.getResponse();
            try {
            	PrintWriter out=response.getWriter();
        		out.print("msg has been received .");
        		out.flush();
        		out.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        	        	
            cSwat.OperatePythonMsg(jsonParam);
        } catch (Exception e) {
            ;
        }
    }
    
    public String CharToJava(String txt) {
		txt = txt.replaceAll("%5B", "[");
		txt = txt.replaceAll("%5D", "]");
		txt = txt.replaceAll("=", "");
//		String regEx = "[\n`~!@#$%^&*()+=|{}':;',\\[\\].<>/?~！@#￥%……&*（）——+|{}【】‘；：”“’。， 、？]";
//		// 可以在中括号内加上任何想要替换的字符，实际上是一个正则表达式
//		Pattern p = Pattern.compile(regEx);
//		Matcher m = p.matcher(txt);// 这里把想要替换的字符串传进来
//		txt = m.replaceAll("").trim();
		
		return txt;
	}
        
	//  @ResponseBody
	  @RequestMapping("/luse")
	  public ModelAndView luse() {
	  	return new ModelAndView("useland.html");
	  }
    
    @ResponseBody
    @RequestMapping(
        value = {"/luse_const"},
        method = {RequestMethod.POST},
        produces = {"application/json;charset=UTF-8"}
    )    
    public String luse_const(@RequestBody String jsonParam) throws IOException {
        System.out.println("luse_const...............................");
        try {
        	jsonParam = URLDecoder.decode(jsonParam);
        	jsonParam = CharToJava(jsonParam);
        	System.out.println(jsonParam);
        	
        	luse _luse = new luse();
			int ymlResult = _luse.InitCalibrate();
			if(ymlResult < 0)
				return JSONArray.toJSONString("yml file is error : application-luse.yml");
			
			//启动
			//if(pythonJson==null)
			//	return JSONArray.toJSONString("it have send to python, but python not result to java");
			//初始化
			if(jsonParam.equals("createlusetopython"))
				return _luse.sendPythonCreateLUse("[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3]");
			else//正常计算传回结果
			{
				int[] luseConvertArrs = _luse.splitLuseLandUser(jsonParam);
				SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMddHHmmss");
				String strTime = sdf.format(Calendar.getInstance().getTime());
							
				String SwatFolder = _luse.ymlSwatFolder;
				String newSwatFolder = SwatFolder + "_" + strTime;
				_api.foldercopy(SwatFolder, 
						newSwatFolder);
				_luse.loadSwat(newSwatFolder);
				
				String json = _luse.swatOperate(luseConvertArrs);
				_luse.sendPythonResultLUse(json);
				return json;
			}
			
        } catch (Exception e) {
        	return JSONArray.toJSONString("failed... exception : ");
        }
    }   
}
