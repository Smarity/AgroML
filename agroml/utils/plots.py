import matplotlib.pyplot as plt
import numpy as np 


from agroml.utils.statistics import *

def plotGraphLinealRegresion(
    x, 
    xName,
    y, 
    yName,
    titleName='',
    fileName='',
    saveFigure=False):    
    '''
    It determines de linear regression and returns a graph with:
     * all the predicted points 
     * the 1:1 line
     * the linear regression line

    Arguments:
        x {1D array} - Measured values.
        xName {str} - Title for abscissa axis.
        y {1D array} - Predicted values.
        yName {str} - Title for ordinate axis
        titleName {str} - Title of the graph.
        fileName {str} - File name.
        saveFigure -> True or False, if True, it saves the figure
    '''

    assert len(x)==len(y)
    x = np.array(x)
    y = np.array(y)

    #Some statistics tests
    mbe = getMeanBiasError(x, y)
    rmse = getRootMeanSquaredError(x, y)
    nse = getNashSuteliffeEfficiency(x, y)


    # linear regresion
    lr_pred = getLinearRegression(x,y)
    lr_text = 'MBE: ' + str(round(mbe[0],4)) + '\n' + 'RMSE: '+str(round(rmse[0],4))+'\n NSE: '+str(round(nse[0],4))
    
    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, lr_pred, color='r', label='LR' )
    ax.plot([0,x[-1]], [0,x[-1]], color='k',label='1:1')
    ax.set(xlabel=xName, ylabel=yName, title=titleName)
    ax.scatter(x, y, label=yName, s=10, facecolors='none',color = 'b')
    ax.text(0.79, 0.25, lr_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')

    plt.legend()
    #ax.grid()
    if saveFigure:
        fig.savefig("fig/"+fileName, format='jpg')

    plt.show()





