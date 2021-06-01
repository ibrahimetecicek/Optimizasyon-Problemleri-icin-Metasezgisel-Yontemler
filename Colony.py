# -*- coding: utf-8 -*-
import math
import numpy as np
import random



def abc_nbr(x, x_index, solution_set):
    delta = 0.5
    x = x.copy()
    i = random.randrange(len(x))
    partner_indices = [ k for k in range(len(solution_set)) ]
    partner_indices.remove(x_index)
    partner_index = random.choice(partner_indices)
    partner = solution_set[partner_index][0]

    # [-delta, delta] aralığında küçük bir gerçek sabit ekleyin
    x[i] = x[i] + delta*(x[i] - partner[i])
    return x

def ColonyAlgo(f, init, nbr, food_source, maxits):
    """Minimize etmek için ABC kodu.
    Şimdiye kadarki en iyi x'i ve f değerini döndürün.


    """
    solution_set = [] # geçmiş oluştur

    best_index = 0
    best_fx = 99999
    for i in range(food_source):
        x = init() # bir ilk rastgele çözüm üret
        fx = f(x)
        solution_set += [[x, fx, 0]]
        if fx < best_fx:
            bestfx = fx
            best_index = i
   
    history = []

    for index in range(1, maxits):

        #İstihdam aşaması
        for i in range(food_source): 

            x = solution_set[i][0]
            fx = solution_set[i][1]
            solution_set[i][2] += 1
            xnew = abc_nbr(x, i , solution_set) # x'in komşusunu oluştur
            fxnew = f(xnew)

            if fxnew < fx:
                solution_set[i][0] = xnew
                solution_set[i][1] = fxnew
                solution_set[i][2] = 0 

                # şimdiye kadarkilerin en iyisini tuttuğumuzdan emin olun
                if fxnew < bestfx:
                    bestfx = fx
                    best_index = i

        #Onlook aşaması

        fitness_sum = sum([sol[1] for sol in solution_set])

        for i in range(food_source): 

            if np.random.random() < (solution_set[i][1]/fitness_sum):

                x = solution_set[i][0]
                fx = solution_set[i][1]
                solution_set[i][2] += 1
                xnew = abc_nbr(x, i , solution_set) # x'in komşusunu oluştur
                fxnew = f(xnew)

                if fxnew < fx:
                    solution_set[i][0] = xnew
                    solution_set[i][1] = fxnew
                    solution_set[i][2] = 0 

                    # şimdiye kadarkilerin en iyisini tuttuğumuzdan emin olun
                    if fxnew < bestfx:
                        bestfx = fx
                        best_index = i

        #Scout phase
        scount_limit = 100
        for i in range(food_source): 
            if solution_set[i][1] > scount_limit and i != best_index:
               
                x = init() # bir ilk rastgele çözüm üret
                fx = f(x)
                solution_set[i][0] = x
                solution_set[i][1] = fx
                solution_set[i][2] = 0
                if fx < bestfx:
                    bestfx = fx
                    best_index = i


        history.append((index,bestfx))
     
    return solution_set[best_index][0], bestfx, history