#Artificial Protozoa Optimizer

#Artificial Protozoa Optimizer

import numpy as np
import random
import time
import os

def APO_func(dim, pop_size, iter_max, Xmin, Xmax, varargin,fobj):
    # Set random seeds
    np.random.seed(int(time.time() * 100) % (2**32 - 1))
    cov=[]
    # Global best
    targetbest = np.array([300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700])
    Fidvec = list(varargin)
    Fid = Fidvec[0]
    runid = Fidvec[1]
    name_convergence_curve = f'APO_Fid_{Fid}_{dim}D.dat'
    f_out_convergence = open(name_convergence_curve, 'a')

    ps = pop_size  # ps denotes protozoa size
    neighbor_pairs = 1  # neighbor_pairs denotes neighbor pairs
    pf_max = 0.1  # pf_max denotes proportion fraction maximum

    # Set points to plot convergence curve
    if runid == 1:
        for i in range(51):  # 51 points to plot
            if i == 0:
                iteration = 1
                f_out_convergence.write(f'iter_F{Fid}\t')
            else:
                iteration = int(iter_max / 50 * i)
            f_out_convergence.write(f'{iteration}\t')
        f_out_convergence.write('\n')

    start_time = time.time()

    protozoa = np.random.rand(ps, dim) * (Xmax - Xmin) + Xmin  # protozoa
    newprotozoa = np.zeros((ps, dim))  # new protozoa
    epn = np.zeros((neighbor_pairs, dim))  # epn denotes effect of paired neighbors

    # Evaluate fitness value
    protozoa_Fit = np.array([fobj(protozoa[i, :])['cost'] for i in range(ps)])
    bestval, bestid = np.min(protozoa_Fit), np.argmin(protozoa_Fit)
    bestProtozoa = protozoa[bestid, :]  # bestProtozoa
    bestFit = bestval  # bestFit
    f_out_convergence.write(f'{runid}\t{bestFit - targetbest[Fid]:.15f}\t')

    # Main loop
    for iter in range( iter_max ):
        protozoa_Fit, index = np.sort(protozoa_Fit), np.argsort(protozoa_Fit)
        protozoa = protozoa[index, :]
        pf = pf_max * random.random()  # proportion fraction
        ri = random.sample(range(ps), int(ps * pf))  # rank index of protozoa in dormancy or reproduction forms
        for i in range(ps):
            if i in ri:  # protozoa is in dormancy or reproduction form
                pdr = 0.5 * (1 + np.cos((1 - i / ps) * np.pi))  # probability of dormancy and reproduction
                if random.random() < pdr:  # dormancy form
                    newprotozoa[i, :] = np.random.rand(1, dim) * (Xmax - Xmin) + Xmin
                else:  # reproduction form
                    flag = random.choice([-1, 1])  # +- (plus minus)
                    Mr = np.zeros((1, dim))  # Mr is a mapping vector in reproduction
                    Mr[0, random.sample(range(dim), int(random.random() * dim))] = 1
                    newprotozoa[i, :] = protozoa[i, :] + flag * random.random() * (Xmin + np.random.rand(1, dim) * (Xmax - Xmin)) * Mr
            else:  # protozoa is foraging form
                f = random.random() * (1 + np.cos(iter / iter_max * np.pi))  # foraging factor
                Mf = np.zeros((1, dim))  # Mf is a mapping vector in foraging
                Mf[0, random.sample(range(dim), int(dim * i / ps))] = 1
                pah = 0.5 * (1 + np.cos(iter / iter_max * np.pi))  # probability of autotroph and heterotroph
                if random.random() < pah:  # protozoa is in autotroph form
                    j = random.randint(0, ps - 1)  # j denotes the jth randomly selected protozoa
                    for k in range(neighbor_pairs):  # neighbor_pairs denotes neighbor pairs
                        if i == 0:
                            km =i
                            kp = i + random.randint(1, ps - i - 1)
                        elif i == ps - 1:
                            km = random.randint(0, i - 1)
                            kp = i
                        else:
                            km = random.randint(0, i - 1)
                            kp = i + random.randint(1, ps - i - 1)
                        wa = np.exp(-np.abs(protozoa_Fit[km] / (protozoa_Fit[kp] + 1e-10)))  # wa denotes weight factor in the autotroph forms
                        epn[k, :] = wa * (protozoa[km, :] - protozoa[kp, :])
                    newprotozoa[i, :] = protozoa[i, :] + f * (protozoa[j, :] - protozoa[i, :] + 1 / neighbor_pairs * np.sum(epn, axis=0)) * Mf
                # else:  # protozoa is in heterotroph form
                #     for k in range(neighbor_pairs):  # neighbor_pairs denotes neighbor pairs
                #         if i == 0:
                #             imk = i
                #             ipk = i + k
                #         elif i == ps - 1:
                #             imk = ps - k
                #             ipk = i
                #         else:
                #             imk = i - k
                #             ipk = i + k
                #         if imk < 0:
                #             imk = 0
                #         elif ipk > ps - 1:
                #             ipk = ps - 1
                #         print("wh",protozoa_Fit[imk] / (protozoa_Fit[ipk] + 1e-10))

                #         wh = np.exp(-np.abs(protozoa_Fit[imk] / (protozoa_Fit[ipk] + 1e-10)))  # wh denotes weight factor in the heterotroph form
                #         print("wh",wh)

                #         epn[k, :] = wh * (protozoa[imk, :] - protozoa[ipk, :])
                #     flag = random.choice([-1, 1])  # +- (plus minus)
                #     Xnear = (1 + flag * random.random() * (1 - iter / iter_max)) * protozoa[i, :]
                #     newprotozoa[i, :] = protozoa[i, :] + f * (Xnear - protozoa[i, :] + 1 / neighbor_pairs * np.sum(epn, axis=0)) * Mf

        newprotozoa = np.clip(newprotozoa, Xmin, Xmax)
        newprotozoa_Fit = np.array([fobj(newprotozoa[i, :])['cost'] for i in range(ps)])
        bin = protozoa_Fit > newprotozoa_Fit
        protozoa[bin, :] = newprotozoa[bin, :]
        protozoa_Fit[bin] = newprotozoa_Fit[bin]
        bestval, bestid = np.min(protozoa_Fit), np.argmin(protozoa_Fit)
        bestProtozoa = protozoa[bestid, :]
        bestFit = bestval
        # if iter % int(iter_max / 50) == 0:
        #     f_out_convergence.write(f'{bestFit - targetbest[Fid]:.15f}\t')
        cov.append(fobj(newprotozoa[i, :])['cost'])
    # recordtime = time.time() - start_time
    # f_out_convergence.write('\n')
    # f_out_convergence.close()
    result=fobj(bestProtozoa)
    return bestProtozoa, bestFit, cov,result    