/**
 * Created with JetBrains WebStorm.
 * User: JINFENGMA
 * Date: 18-11-15
 * Time: 上午11:08
 * To change this template use File | Settings | File Templates.
 */
/**
 * Created with JetBrains WebStorm.
 * User: JINFENGMA
 * Date: 18-6-22
 * Time: 上午10:17
 * To change this template use File | Settings | File Templates.
 */
//基础数据
var ip = "localhost:8080";
var info = null;
$(document).ready(function() {
	initinfo() ;
});//61222833
Date.prototype.Format = function (fmt) { //author: meizz
    var o = {
        "M+": this.getMonth() + 1, //月份
        "d+": this.getDate(), //日
        "h+": this.getHours(), //小时
        "m+": this.getMinutes(), //分
        "s+": this.getSeconds(), //秒
        "q+": Math.floor((this.getMonth() + 3) / 3), //季度
        "S": this.getMilliseconds() //毫秒
    };
    if (/(y+)/.test(fmt)) fmt = fmt.replace(RegExp.$1, (this.getFullYear() + "").substr(4 - RegExp.$1.length));
    for (var k in o)
        if (new RegExp("(" + k + ")").test(fmt)) fmt = fmt.replace(RegExp.$1, (RegExp.$1.length == 1) ? (o[k]) : (("00" + o[k]).substr(("" + o[k]).length)));
    return fmt;
}


function sencreate_sa() {
    var url = "http://"+ip+"/sencreate_sa";    
    var config = {
	    async:false,
	    xhrFields: {
	        withCredentials: true
	    },
	    url: url,
	    type: "POST",
	    beforeSend: function () {
	    	ctrldisable();
	    },
	    success: function(result) {
	        alert(result);
	        ctrlenable()
	    },
	    error: function(r){
	        alert('已发送，返回错误');
	        ctrlenable()
	    },
	    failed: function(r){
	        alert('failed');
	        ctrlenable()
	    }
	};
	$.ajax(config);
};

function sencreate_bo() {
    var url = "http://"+ip+"/sencreate_bo";    
    var config = {
	    async:false,
	    xhrFields: {
	        withCredentials: true
	    },
	    url: url,
	    type: "POST",
	    beforeSend: function () {
	    	ctrldisable();
	    },
	    success: function(result) {
	        alert(result);
	        ctrlenable()
	    },
	    error: function(r){
	        alert('已发送，返回错误');
	        ctrlenable()
	    },
	    failed: function(r){
	        alert('failed');
	        ctrlenable()
	    }
	};
	$.ajax(config);
};

function resultdata() {
    var url = "http://"+ip+"/resultdata";
    var datap = {
    		operate: document.getElementById('operate').value,
    		engine: document.getElementById('engine').value,
    		projectname: document.getElementById('projectname').value,
    		calibtype: document.getElementById('calibtype').value,
    		algo: document.getElementById('algo').value
    }
    var config = {
	    async:false,
	    xhrFields: {
	        withCredentials: true
	    },
	    url: url,
	    data: datap,
        dataType:'json',
	    type: "POST",
	    beforeSend: function () {
	    	ctrldisable();	    	 
	    },
	    success: function(result) {
	        var str = '共有' + result.length + '条记录, ';
	        str += '数据量超大，请参见服务端工程目录下 (c:/calibrate/ ' + info.calibAlgos[0] + '.txt)';
	        document.getElementById('jsontext').value = str;
	        ctrlenable()
	    },
	    error: function(r){
	    	document.getElementById('jsontext').value = "";
	        alert('已发送，返回错误');
	        ctrlenable()
	    },
	    failed: function(r){
	        alert('failed');
	        ctrlenable()
	    }
	};
	$.ajax(config);
};

function initinfo() {
    var url = "http://"+ip+"/initinfo";
    
    var config = {
	    async:false,
	    xhrFields: {
	        withCredentials: true
	    },
	    url: url,
        dataType:'json',
	    type: "POST",
	    beforeSend: function () {
	    	ctrldisable();
	    },
	    success: function(result) {
	    	info = result;
	    	document.getElementById('operate').value = 'calibrate';
    		document.getElementById('engine').value = result.engine;
    		document.getElementById('projectname').value = result.swatProjectName;
    		document.getElementById('calibtype').value = result.calibtype;
    		document.getElementById('algo').value = result.calibAlgos[0];
    		ctrlenable()
	    },
	    error: function(r){
	    	ctrlenable()
	    },
	    failed: function(r){
	    	ctrlenable()
	    }
	};
	$.ajax(config);
};

function ctrlenable()
{
	var b = false;
	document.getElementById('sencreate').disable = b;
	document.getElementById('sencreate2').disable = b;
	document.getElementById('resultdata').disable = b;
}
function ctrldisable()
{
	var b = true;
	document.getElementById('sencreate').disable = b;
	document.getElementById('sencreate2').disable = b;
	document.getElementById('resultdata').disable = b;
}
