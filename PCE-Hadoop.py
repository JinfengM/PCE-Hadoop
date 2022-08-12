# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:37:25 2022

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 09:49:38 2021

@author: JinfengMa jfma@rcees.ac.cn
"""


import pandas as pd
import numpy as np
import chaospy
from matplotlib.pyplot import MultipleLocator
from datetime import datetime
#date label
x=['2010-1','2010-2','2010-3','2010-4','2010-5','2010-6','2010-7','2010-8','2010-9','2010-10','2010-11','2010-12','2011-1','2011-2','2011-3','2011-4','2011-5','2011-6','2011-7','2011-8','2011-9','2011-10','2011-11','2011-12','2012-1','2012-2','2012-3','2012-4','2012-5','2012-6','2012-7','2012-8','2012-9','2012-10','2012-11','2012-12','2013-1','2013-2','2013-3','2013-4','2013-5','2013-6','2013-7','2013-8','2013-9','2013-10','2013-11','2013-12','2014-1','2014-2','2014-3','2014-4','2014-5','2014-6','2014-7','2014-8','2014-9','2014-10','2014-11','2014-12']
#这个NSE和Metric的R2一样
#define metrics
def nse(predictions, targets):
    return (1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2)))

def R2(predictions, targets):
    temp1=np.sum((targets-np.mean(targets))*(predictions-np.mean(predictions)))
    #print(temp1)
    temp2=np.sum((targets-np.mean(targets))*(targets-np.mean(targets)))
    #print(temp2)
    temp3=np.sum((predictions-np.mean(predictions))*(predictions-np.mean(predictions)))
    #print(temp3)
    return(temp1**2)/(temp2*temp3)

def pbis(predictions, targets):
    temp1=np.sum(predictions-targets)/np.sum(targets)
    return temp1

def MaxMinNormalization(x,Min,Max):
    x = Min+x*(Max-Min)
    return x

def RevMaxMinNormalization(x,Max,Min):
    x = (x-Min)/(Max-Min)
    return x
#################################Training PCE#######################################
filename='6561'
input_csv=pd.read_csv(filename+'.csv',header=None)
#print(input_csv.shape[0])
paras_list=[i for i in range(8)]
#print(input_csv.iloc[:,paras_list])

#获取参数集（60000,8）
pd_paras_list=input_csv.iloc[:,paras_list]

sim1_list=[i+9 for i in range(60)]
#print(input_csv.iloc[:2,sim1_list])
pd_sim1_list=input_csv.iloc[:,sim1_list]

obs1_list=[i+70 for i in range(60)]
#取出1个观测值，观测值是重复模式
pd_obs1_list=input_csv.iloc[:1,obs1_list]


sim2_list=[i+131 for i in range(60)]
#print(input_csv.iloc[:2,sim2_list])
pd_sim2_list=input_csv.iloc[:,sim2_list]

obs2_list=[i+192 for i in range(60)]
pd_obs2_list=input_csv.iloc[:2,obs2_list]

sim3_list=[i+253 for i in range(60)]
#print(input_csv.iloc[:2,sim3_list])
pd_sim3_list=input_csv.iloc[:,sim3_list]

obs3_list=[i+314 for i in range(60)]
pd_obs3_list=input_csv.iloc[:2,obs3_list]


#定义分布
#define distribution :default Uniform
CANMX=chaospy.Uniform(0,100)
CN2=chaospy.Uniform(35,98)
CH_N2=chaospy.Uniform(0,0.3)
CH_K2=chaospy.Uniform(5.0,130)
ALPHA_BNK=chaospy.Uniform(0,1.0)
SOL_AWC=chaospy.Uniform(0,1)
SOL_K=chaospy.Uniform(0,2000)
SOL_BD=chaospy.Uniform(0.9,2.5)

#定义联合分布
#define joint distribution
joint=chaospy.J(CANMX,CN2,CH_N2,CH_K2,ALPHA_BNK,SOL_AWC,SOL_K,SOL_BD)
#joint=chaospy.Iid(chaospy.Normal(0,1), 8)
#从正交积分中产生节点和权重，节点用于评估
#gauss_quad= chaospy.generate_quadrature(2,joint,rule='gaussian')
#nodes,weights=gauss_quad

#定义正交多项式扩展
#orth_ttr

#expansion = chaospy.expansion.stieltjes(2, joint)
expansion = chaospy.generate_expansion(2, joint, rule='cholesky')
#print(expansion)
#真实模型计算，产生输入-输出响应关系
evaluations=pd_sim1_list

#估计傅里叶系数，产生近似模型
#point collocation
#得到采样参数集（8,60000）
#nodes=joint.sample(60000,rule="hammersley",seed=1234)
nodes=pd_paras_list.T
#之前方差误差过大的原因：nodes和evaluations不配套
approx_solver = chaospy.fit_regression(expansion, nodes, evaluations)

#计算均值和方差
mean=chaospy.E(approx_solver,joint)
deviation=chaospy.Std(approx_solver,joint)
#MC验证

#define MC data
number=10000
# if number==6561 strtraining='Training'
# else
# strtraining='Validataion'
strtraining='Training'
newfilename=str(number)
newinput_csv=pd.read_csv(newfilename+'.csv',header=None)

#获取参数集合
newparas_list=[i for i in range(8)]
#print(newinput_csv.iloc[:,newparas_list])
newpd_paras_list=newinput_csv.iloc[:,newparas_list]

newsim1_list=[i+9 for i in range(60)]
#print(input_csv.iloc[:2,sim1_list])
newpd_sim1_list=newinput_csv.iloc[:,newsim1_list]

mcobsmean=[]
mcobsvar=[]
mcvalues=[]
for item in range(newpd_sim1_list.shape[1]):
    a=newpd_sim1_list.iloc[:,[item]].mean().tolist()#均值
    b=newpd_sim1_list.iloc[:,[item]].var().tolist()#方差
    mcvalues.append(newpd_sim1_list.iloc[:,[item]].values)#MC模拟值
    mcobsmean.append(a)
    mcobsvar.append(b)
mcobsmean=list(np.array(mcobsmean).flat)
mcobsvar=list(np.array(mcobsvar).flat)
#(1)uniform(a,b)
newnodes=newpd_paras_list.T.values
#(2)uniform(a,b)->uniform(0,1)
#得到MC模拟值
newapprox_evaluations=approx_solver(*newnodes)
simmean=[]
simvar=[]
simvalues=[]
for item in range(newapprox_evaluations.shape[0]):
    simmean.append(newapprox_evaluations[item].mean())#均值
    simvar.append(newapprox_evaluations[item].var())#方差
    simvalues.append(newapprox_evaluations[item])#PCE近似值
    

from sklearn import linear_model        #表示，可以调用sklearn中的linear_model模块进行线性回归。
import numpy as np
model = linear_model.LinearRegression()
model.fit(np.array(simvar).reshape(-1,1), np.array(mcobsvar).reshape(-1,1))
a=model.intercept_
b=model.coef_
#newY=a+b*x
#生产拟合曲线
newY=a+b*simvar
newY=newY.flatten()

print(model.score(np.array(simvar).reshape(-1,1),np.array(mcobsvar).reshape(-1,1)))

import matplotlib
from matplotlib.font_manager import _rebuild
_rebuild()
#################################月均值时间序列#######################################
import matplotlib.pyplot as plt
with plt.style.context(['science', 'no-latex']):
    #matplotlib.rcParams['font.family'] ='sans-serif'
    #matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    #matplotlib.rcParams['axes.unicode_minus']=False
    fig, ax = plt.subplots()
    ax.plot(x,simmean,label='PCE',color='#0C5DA5',linewidth=0.5)
    ax.scatter(x,mcobsmean,label='MC',color='#FF2C00',s=0.5)
    x_major_locator=MultipleLocator(12)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.legend()
    ax.set_title(strtraining)
    ax.set_ylabel('Mean monthly flow(m$^3$/s)')
    ax.set_xlabel('Date')
    ax.autoscale(tight=True)
    fig.savefig('figures/'+str(number)+'-pce-mean'+newfilename+'.png', dpi=300)
    
#################################月方差时间序列#######################################
with plt.style.context(['science', 'no-latex']):
    #matplotlib.rcParams['font.family'] ='sans-serif'
    #matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    #matplotlib.rcParams['axes.unicode_minus']=False
    fig, ax = plt.subplots()
    ax.plot(x,simvar,label='PCE',color='#0C5DA5',linewidth=0.5)
    #ax.plot(x,mcobsvar,label='mcobsvar',color='#FF2C00',linewidth=0.5)
    ax.scatter(x,mcobsvar,label='MC',color='#FF2C00',s=0.5)
    x_major_locator=MultipleLocator(12)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_ylim([0,4500])
    ax.legend()
    ax.set_title(strtraining)
    #ax.legend(title='variance方差')
    ax.set_ylabel('Flow variance(m$^3$/s)')
    #ax.autoscale(tight=True)
    fig.savefig('figures/'+str(number)+'-pce-var'+newfilename+'.png', dpi=300)
    
#绘制对比图
#################################均值点对点对比#######################################
with plt.style.context(['science', 'no-latex']):
    #matplotlib.rcParams['font.family'] ='sans-serif'
    #matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    #matplotlib.rcParams['axes.unicode_minus']=False
    fig, ax = plt.subplots()
    ax.plot(simmean,simmean,label='PCE',color='#0C5DA5',linewidth=0.5)
    ax.scatter(simmean,mcobsmean,label='MC',color='#FF2C00',s=0.5)
    x_major_locator=MultipleLocator(50)
    ax.xaxis.set_major_locator(x_major_locator)
    #ax.legend()
    ax.set_title(strtraining)
    ax.set_ylabel('Mean monthly flow using MC(m$^3$/s)')
    ax.set_xlabel('Mean monthly flow using PCE(m$^3$/s)')

    ax.autoscale(tight=True)
    fig.savefig('figures/'+str(number)+'-mean-compare'+newfilename+'.png', dpi=300)
    
    #################################方差点对点对比#######################################
with plt.style.context(['science', 'no-latex']):
    #matplotlib.rcParams['font.family'] ='sans-serif'
    #matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    #matplotlib.rcParams['axes.unicode_minus']=False
    fig, ax = plt.subplots()
    ax.plot(simvar,simvar,label='Line with unit slope',color='#0C5DA5',linewidth=0.5)
    #R2 is calculated by R2(simvar,mcobsvar)
    ax.plot(simvar,newY,label='Best fit R$^2$=0.99',color='#FF2C00',linewidth=0.5)
    ax.scatter(simvar,mcobsvar,color='#FF2C00',s=0.5)
    x_major_locator=MultipleLocator(1000)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(x_major_locator)
    ax.legend()
    ax.set_title(strtraining)
    ax.set_xlim([0,4500])
    ax.set_ylim([0,4500])
    ax.set_ylabel('Flow variance using MC(m$^3$/s)')
    ax.set_xlabel('Flow variance using PCE(m$^3$/s)')
    #ax.autoscale(tight=True)
    fig.savefig('figures/'+str(number)+'-var-compare'+newfilename+'.png', dpi=300)
    

#利用PCE进行敏感性分析
CANMX=chaospy.Uniform(0,100)
CN2=chaospy.Uniform(35,98)
CH_N2=chaospy.Uniform(0,0.3)
CH_K2=chaospy.Uniform(5.0,130)
ALPHA_BNK=chaospy.Uniform(0,1.0)
SOL_AWC=chaospy.Uniform(0,1)
SOL_K=chaospy.Uniform(0,2000)
SOL_BD=chaospy.Uniform(0.9,2.5)
    
import SALib.sample.saltelli
from SALib.analyze import sobol
from SALib.analyze import morris
import SALib.analyze.fast
import SALib.analyze.delta
import SALib.analyze.dgsm
import SALib.analyze.rbd_fast

problem = {
'num_vars': 8,
'names': ['CANMX', 'CN2', 'CH_N2','CH_K2','ALPHA_BNK','SOL_AWC','SOL_K','SOL_BD'],
'bounds': [[0, 100],
           [35, 98],
           [0, 0.3],
           [5, 130],
           [0, 1],
           [0, 1],
           [0, 2000],
           [0.9, 2.5]
           ]
}
#sobol敏感性分析
timebefore=datetime.now()

param_values = SALib.sample.saltelli.sample(problem, 1024)
sanodes=param_values.T
#产生PCE预测时间序列
PCEevaluation=approx_solver(*sanodes)
#NSE计算敏感指数
PCEevaluation=PCEevaluation.T
nseResult=[]
#PCEevaluation.shape[0]
for item in range(PCEevaluation.shape[0]):
    #计算与实测值
    predictions=np.array(PCEevaluation[item].flat)
    targets=np.array(pd_obs1_list)
    result=nse(predictions,targets)
    #result=nse(targets,predictions)
    nseResult.append(result)
Y=np.array(nseResult)
Siso = sobol.analyze(problem, Y)
timeafter=datetime.now()
timecost=timeafter-timebefore

#morris 敏感性分析
import SALib.sample.morris
param_values =SALib.sample.morris.sample(problem, N=1024, num_levels=4,optimal_trajectories=None)
monodes=param_values.T
#产生morris预测时间序列
PCEevaluation=approx_solver(*monodes)
#NSE计算敏感指数
PCEevaluation=PCEevaluation.T
nseResult=[]
#PCEevaluation.shape[0]
for item in range(PCEevaluation.shape[0]):
    #计算与实测值
    predictions=np.array(PCEevaluation[item].flat)
    targets=np.array(pd_obs1_list)
    result=nse(predictions,targets)
    #result=nse(targets,predictions)
    nseResult.append(result)
Y=np.array(nseResult)
Simo = morris.analyze(problem,param_values, Y)

#FAST敏感性分析
import SALib.sample.fast_sampler
param_values = SALib.sample.fast_sampler.sample(problem, 1024)
fanodes=param_values.T
#产生PCE预测时间序列
PCEevaluation=approx_solver(*fanodes)
#NSE计算敏感指数
PCEevaluation=PCEevaluation.T
nseResult=[]
#PCEevaluation.shape[0]
for item in range(PCEevaluation.shape[0]):
    predictions=np.array(PCEevaluation[item].flat)
    targets=np.array(pd_obs1_list)
    result=nse(predictions,targets)
    #result=nse(targets,predictions)
    nseResult.append(result)
Y=np.array(nseResult)
Sifa =SALib.analyze.fast.analyze(problem, Y)

#Delta敏感性分析
import SALib.sample.latin
param_values = SALib.sample.latin.sample(problem, 1024)
denodes=param_values.T
#产生PCE预测时间序列
PCEevaluation=approx_solver(*denodes)
#NSE计算敏感指数
PCEevaluation=PCEevaluation.T
nseResult=[]
#PCEevaluation.shape[0]
for item in range(PCEevaluation.shape[0]):
    predictions=np.array(PCEevaluation[item].flat)
    targets=np.array(pd_obs1_list)
    result=nse(predictions,targets)
    #result=nse(targets,predictions)
    nseResult.append(result)
Y=np.array(nseResult)
Side =SALib.analyze.delta.analyze(problem,param_values, Y)

#DGSM敏感性分析
import SALib.sample.finite_diff
param_values = SALib.sample.finite_diff.sample(problem, 1024)
dgnodes=param_values.T
#产生PCE预测时间序列
PCEevaluation=approx_solver(*dgnodes)
#NSE计算敏感指数
PCEevaluation=PCEevaluation.T
nseResult=[]
#PCEevaluation.shape[0]
for item in range(PCEevaluation.shape[0]):
    predictions=np.array(PCEevaluation[item].flat)
    targets=np.array(pd_obs1_list)
    result=nse(predictions,targets)
    #result=nse(targets,predictions)
    nseResult.append(result)
Y=np.array(nseResult)
Sidg =SALib.analyze.dgsm.analyze(problem,param_values, Y)

#rbd-fast敏感性分析
import SALib.sample.latin
param_values = SALib.sample.latin.sample(problem, 1024)
rbnodes=param_values.T
#产生PCE预测时间序列
PCEevaluation=approx_solver(*rbnodes)
#NSE计算敏感指数
PCEevaluation=PCEevaluation.T
nseResult=[]
#PCEevaluation.shape[0]
for item in range(PCEevaluation.shape[0]):
    predictions=np.array(PCEevaluation[item].flat)
    targets=np.array(pd_obs1_list)
    result=nse(predictions,targets)
    #result=nse(targets,predictions)
    nseResult.append(result)
Y=np.array(nseResult)
Sirb =SALib.analyze.rbd_fast.analyze(problem,param_values, Y)
