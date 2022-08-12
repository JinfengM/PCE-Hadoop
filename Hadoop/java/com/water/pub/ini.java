package com.water.pub;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ini {
    public ini() {
    }

    public static String getIni(String file,String section,String variable,String defaultValue)  
			 throws IOException 
	{  
		String strLine, value = "";
     BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
     boolean isInSection = false;
     try {
         while ((strLine = bufferedReader.readLine()) != null) {
             strLine = strLine.trim();
             strLine = strLine.split("[;]")[0];
             Pattern p;
             Matcher m;
             p = Pattern.compile("\\[\\w+]");//Pattern.compile("file://[//s*.*//s*//]");
             m = p.matcher((strLine));
             if (m.matches()) {
                 p = Pattern.compile("\\[" + section + "\\]");//Pattern.compile("file://[//s*" + section + "file://s*//]");
                 m = p.matcher(strLine);
                 if (m.matches()) {
                     isInSection = true;
                 } else {
                     isInSection = false;
                 }
             }
             if (isInSection == true) {
                 strLine = strLine.trim();
                 String[] strArray = strLine.split("=");
                 if (strArray.length == 1) {
                     value = strArray[0].trim();
                     if (value.equalsIgnoreCase(variable)) {
                         value = "";
                         return value;
                     }
                 } else if (strArray.length == 2) {
                     value = strArray[0].trim();
                     if (value.equalsIgnoreCase(variable)) {
                         value = strArray[1].trim();
                         return value;
                     }
                 } else if (strArray.length > 2) {
                     value = strArray[0].trim();
                     if (value.equalsIgnoreCase(variable)) {
                         value = strLine.substring(strLine.indexOf("=") + 1).trim();
                         return value;
                     }
                 }
             }
         }
     } finally {
         bufferedReader.close();
     }
     return defaultValue;
	}  

    public static void main(String[] args) {
    }
}