import numpy as np
from scipy.interpolate import interp1d

def find_roots(func, rng, var_space, res=1000, prnt=False, log = False):
    
    n = np.linspace(rng[0],rng[1],res)
    evar = np.log10(func(n))
    
    roots_list = []
    for t in var_space:
        intersec_list = []
        for i in range(res-1):

            if (t > evar[i] and  t < evar[i+1]) or (t < evar[i]  and t > evar[i+1]):
                slope = ((n[i+1]-n[i])/(evar[i+1]-evar[i]))
                x_int = n[i]+(t-evar[i])*slope

                if prnt:
                    print('------------------------------------')
                    print('Y_G: ',t)
                    print('Y: ',evar[i],evar[i+1])
                    print('X: ',n[i],n[i+1])
                    print('X_int: ',x_int)
                    print('Slope: ',slope)

                intersec_list.append([x_int,slope])
        roots_list.append(intersec_list)
    return roots_list

def compute_p_TS(err_space, var_space, x_space, res, prob):

    A4 = np.zeros([len(err_space),len(var_space)])
    
    for i,s in enumerate(err_space):
        for j,t in enumerate(var_space):

            p = 0
            z_list = find_roots(prob.var_XgZ, x_space,[t],3*res,prnt=False,log = True)[0]
            for zsl in z_list:
                z = zsl[0]
                sl = zsl[1]
                x = [np.sqrt(10**s)+prob.exp_XgZ(z),-np.sqrt(10**s)+prob.exp_XgZ(z)]
                p+=(prob.p_XZ(np.array([[x[0]],[z]]))+prob.p_XZ(np.array([[x[1]],[z]])))*np.abs(0.5*np.sqrt(10**s)*np.log(10)*sl)

            A4[i,j] = p
    return A4


def get_equidistant_quantile(err,var,res=1e2,quantile = 0.95,clean_outliers = False, err_range = [],var_range=[],corrected = False,):

    var = np.log10(var + 1e-15).reshape(-1)
    err = np.log10(err + 1e-15).reshape(-1)

    if clean_outliers:
        if len(err_range) != 2:
            err_range = [np.quantile((err),clean_outliers), np.quantile((err),1-clean_outliers)]
        else:
            err_range = np.log10(err_range)

        if len(var_range) != 2:
            var_range = [np.quantile((var),clean_outliers), np.quantile((var),1-clean_outliers)]
        else:
            var_range = np.log10(var_range)

        clean_idx_var = np.where(np.logical_and(var >= var_range[0], var <= var_range[1]))
        clean_idx_err = np.where(np.logical_and(err >= err_range[0], err <= err_range[1]))
        clean_ixd = np.intersect1d(clean_idx_var, clean_idx_err)
        err = err[clean_ixd]
        var = var[clean_ixd]

    bin_edges = np.linspace(np.min(var) - 1e-4, np.max(var) + 1e-4, res + 1)
    bin_quantiles = []

    for i in range(res):
        tmp_idzs = np.where(np.logical_and(var > bin_edges[i], var < bin_edges[i + 1]))[0]
        if np.size(tmp_idzs) >= 1:
            if corrected:
                quantile_tmp = np.clip((1 + 1 / len(tmp_idzs)) * quantile,0,1)
                quant_idx =np.clip(int(np.ceil(len(tmp_idzs) * quantile_tmp)), 0, len(tmp_idzs))
            else:
                quant_idx = int(np.ceil(len(tmp_idzs) * quantile))

            bin_quantiles.append(np.sort(err[tmp_idzs])[quant_idx - 1])
        else:
            bin_quantiles.append(bin_quantiles[-1])


    bin_quantiles = np.array(bin_quantiles)
    draw_edges = np.array(2 * [bin_edges]).T.reshape(-1)[1:-1]
    draw_quantiles = np.array(2 * [bin_quantiles]).T.reshape(-1)

    f = interp1d(draw_edges, draw_quantiles,fill_value="extrapolate")
    return f