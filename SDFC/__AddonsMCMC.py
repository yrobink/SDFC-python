# -*- coding: utf-8 -*-

## Copyright(c) 2020 / 2023 Occitane Barbaux and Yoann Robin
## 
## This file is part of SDFC.
## 
## SDFC is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## SDFC is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with SDFC.  If not, see <https://www.gnu.org/licenses/>.


##############
## Packages ##
##############

import numpy as np
import scipy.stats as sc
import os
import arviz 
from tabulate import tabulate

import matplotlib.backends.backend_pdf as mpdf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.integrate as si
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd


##############
## Functions ##
##############

def Summary_run_table(ns_law,prior_law,MLE_theta=["","",""],pathOut="",q_l=0.05,q_h=0.95,show=True):
    coef=ns_law._lhs.names
    accept=ns_law.info_.accept
    draw=ns_law.info_.draw
    true_theta=prior_law.mean
    n_mcmc_drawn=len(accept)
    accept_all= np.sum(accept) / n_mcmc_drawn
    if len(accept.shape)>1:
        #Hybrid
        accept_all= np.sum(accept) / (n_mcmc_drawn*draw.shape[1])
        accept_para=accept.sum(axis=0)/n_mcmc_drawn

    else:
        accept_all= np.sum(accept) / n_mcmc_drawn
        accept_para=[accept_all]*draw.shape[1]
        
        #add effective sample
    
    #effective_samples_around=multiESS(draw, b='sqroot', Noffsets=10, Nb=None)
    idata = arviz.convert_to_inference_data(np.expand_dims(draw, 0))
    effective_samples_para=arviz.ess(idata).x.to_numpy()
    #add effective sample

    mean_para=np.mean(draw, axis = 0 )
    med_para=np.median(draw, axis = 0 )
    qlow_para=np.quantile(draw,q_l, axis = 0 )
    qhigh_para=np.quantile(draw,q_h, axis = 0 )
    if show:
        print("Total acceptation rate is "+ str(accept_all)+" for "+str(n_mcmc_drawn) +" iterations")
       # print("Total number of effective samples is "+ str(effective_samples_around))
    a = []#np.empty(shape=(draw.shape[1], 7))
    for i in range(draw.shape[1]):
        #add name
        a.append([coef[i], true_theta[i],MLE_theta[i],qlow_para[i],med_para[i],mean_para[i],qhigh_para[i],effective_samples_para[i],accept_para[i]])
        colnames=["Paramètre", "Prior","MLE","Q_"+str(q_l), "Médiane", "Moyenne","Q_"+str(q_h), "eff sample" ,"acceptation rate"]

        
    table=tabulate(a, headers=colnames, tablefmt='fancy_grid')
    if show:
        print(table)
    
    with open(os.path.join( pathOut ,"Table_MCMC.txt"), "w") as outf:
        outf.write("Total acceptation rate is "+ str(accept_all)+" for "+str(n_mcmc_drawn) +" iterations\n")
        #outf.write("Total number of effective samples is "+ str(effective_samples_around)+"\n")
        outf.write(table)
        
        
def Para_Runs(ns_law,prior_law,MLE_theta=["","",""],pathOut="",q_l=0.05,q_h=0.95,show=True):
    coef=ns_law._lhs.names
    accept=ns_law.info_.accept
    draw=ns_law.info_.draw
    prior_mean=prior_law.mean
    n_mcmc_drawn=len(accept)
    
    iteration=list(range(len(draw)))
    ofile=os.path.join( pathOut,"ParaComp_MCMC.pdf" )
    pdf = mpdf.PdfPages( ofile )
    for i in range(len(coef)):
        fig = plt.figure( figsize = (20,10) )
        gs = GridSpec(2, 2, figure=fig)
        ax = fig.add_subplot( gs[0, 0] )
        plt.plot(iteration,draw[:,i])
        plt.xlabel("Iterations")
        plt.ylabel(coef[i])
        
        if MLE_theta[i]!="":
            plt.axhline(y = MLE_theta[i], color = 'm', linestyle = '-',label="MLE")
            names=["Iterations","MLE","Prior mean"]
        else:
            names=["Iterations","Prior mean"]
        plt.axhline(y = prior_mean[i], color = 'g', linestyle = '-',label="prior mean")
        ax.legend(names,fontsize = 15)
     
        #plt.show()
        
        ax = fig.add_subplot(gs[0, 1])
        
        prior_loc0=prior_law.rvs(10000)[:,i]
        ker_b=sc.gaussian_kde(prior_loc0)
        posterior_loc0=draw[:,i]
        ker_a=sc.gaussian_kde(posterior_loc0)
        def y_pts(pt):
            y_pt = min(ker_a(pt), ker_b(pt))
            return y_pt
        # Store overlap value.
        overlap = si.quad(y_pts,min(posterior_loc0),max(posterior_loc0)) 
        overlap_per=round(overlap[0]*100,2)        
        

        sns.kdeplot(np.array(prior_loc0),color='g',fill=True)
        sns.kdeplot(np.array(posterior_loc0),color='b')
        ax=sns.histplot(np.array(posterior_loc0),stat="density")
        if MLE_theta[i]!="":
            plt.axvline(x= MLE_theta[i], color = 'm', linestyle = '-')
        plt.axvline(x= prior_mean[i], color = 'g', linestyle = '-')
        plt.axvline(x= np.median(draw[:,i]), color='black',linestyle = '-')
        plt.xlim(min(posterior_loc0),max(posterior_loc0))
        ax.text(0.80, 0.98, "Overlap: \n"+str(overlap_per)+"%", ha="left", va="top", transform=ax.transAxes)
        #plt.show()

        #pdf.savefig(fig)
        #plt.close(fig) 
        ax = fig.add_subplot( gs[1, :])
        plot_acf(draw[:,i],lags=120,ax=ax )
        plt.ylim([-0.1,1.1])
        plt.title("Autocorrelation " +coef[i])
        if show:
            plt.show()
        pdf.savefig(fig)
        plt.close(fig)
    swarm_plot = sns.pairplot(pd.DataFrame(draw, columns = coef),corner=True,kind="kde")
    fig = swarm_plot.fig
    pdf.savefig(fig)
    pdf.close()
   

