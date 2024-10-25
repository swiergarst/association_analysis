import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def plot_gof(val_dict, centers = ["Leiden", "Rotterdam", "Maastricht"], central = False, weighted = True):
    loss = np.array(val_dict['mae'])
    n_rounds = loss.shape[1]
    

    if central:
        if weighted:
            weights = val_dict['sizes']
            central_loss = np.sum(np.array([ loss[0, : , i] * weights[i] for i in range(loss.shape[2])]), axis = 0)
            central_loss /= np.sum(weights)
        else:
            central_loss = np.mean(loss, axis = 2)[0,:]
        
        plt.plot(np.arange(n_rounds), central_loss, label = "all")
    else:
        for i in range(loss.shape[2]):
            plt.plot(np.arange(n_rounds), loss[0,:,i], label = centers[i])
        
    plt.title('Goodness of fit per iteration, ' + val_dict['model'])
    plt.legend()
    plt.grid()
    plt.xlabel("iteration")
    plt.ylabel("MAE")

def weighted_mean(data, sizes, selec="HPC"):

    if (selec == "HPC"):
        print(data.shape)
        wm = np.zeros((data.shape[1], data.shape[2]))
        #dset_sizes = np.array([3000, 292, 935])

        for i in range(data.shape[0]):
            wm += sizes[i] * data[i,:,:]

        wm /= np.sum(sizes) 
        return(wm)
    elif (selec == "V6"):
        wm = np.zeros((data.shape[0], data.shape[2]))
        #dset_sizes = np.array([3000, 292, 935])

        for i in range(data.shape[1]):
            wm += sizes[i] * data[:,i,:]

        wm /= np.sum(sizes) 
        return(wm)



def plot_range(data, data2=None, line='-', color=None, label = None, convert = False):
    if data2 == None:
        x = np.arange(data.shape[1])
    else:
        x = np.mean(data2, axis =0)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    #std = np.var(data, axis=0)
    if color==None:
        ax = plt.plot(x, mean,line, label=label)
        plt.fill_between(x, mean-std, mean+std, alpha = 0.5)
    else:
        if convert:
            r,g,b,a = color.to_rgba(color, alpha = 0.5)
#alpha = 0.5
            r_new = ((1 - a) * 1) + (a * r)
            g_new = ((1 - a) * 1) + (a * g)
            b_new = ((1 - a) * 1) + (a * b)

            ax = plt.plot(x, mean, line, color=color, label=label)
            plt.fill_between(x, mean-std, mean+std, color = (r_new, g_new, b_new))
        else:
            ax = plt.plot(x, mean,line, color=color, label = label)
            plt.fill_between(x, mean-std, mean+std, alpha = 0.5, color=color)

    return ax



def calc_significance(val_dict,best_model, name, n_tot):
    name_ind = val_dict['coef_names'].index(name)
    beta = val_dict['all_global_betas'][best_model][name_ind]
    p_val = st.t.sf(abs(beta / val_dict['standard_error'][name_ind]), df = n_tot - 2)
    return p_val