import os
import gamerunner as gr
import gamehandler as gh

def static(prechoice=None):
    os.system("clear")
    if prechoice is not None:
        choice = prechoice
    print("\n")
    if prechoice is None:
        choice = int(input("Enter the desired mode :\n0-Quit\n1-regMul\n2-polyMul\n3-RegDet\n4-PolyDet\n5-polyEval\n6-evalRoot\n7-evalRootPoly\n8-surdGame\n9-divGame\n10-polyDiv\n11-EigenGame\n12-RootGame\n13-DiscGame\n14-PFD\n15-IntegralGame\n16-RegDig\n17-Fourier Series\n18-Equation system\n19-Mean\n20-Stdev\n21-diffeq\n22-curvatureGame\n23-TGame\n24-LineIntegralGame\n25-DiverganceGame\n26-LineIntegralSc\n27-Shuffle\n28-Calc Suite\n29-Arithmetic Suite\n"))
    if choice == 1:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : ")) 
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        float_mode = int(input("Float mode ? (1 yes/0 no)"))
        a_fl = 0
        if float_mode:
            a_fl = int(input("Digits after floating point : "))
        
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "float_mode" : float_mode, "ndigits" : a_fl}
        stats = gr.general_runner(gh.regMul, rounds, inpt_dict, md)#multgame.regMulGame(number_of_rounds=rounds, nrange=ranges, float_mode=float_mode, after_float_point=a_fl)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 2:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        max_deg = int(input("Maximum degree : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : max_deg}
        stats = gr.general_runner(gh.polyMul, rounds, inpt_dict, md)#multgame.polyMulGame(number_of_rounds=rounds, max_deg=max_deg, nrange=ranges[:])
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 3:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        dims = int(input("Dim : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "dim" : dims}
        stats = gr.general_runner(gh.regDet, rounds, inpt_dict, md)#matrixgames.regDetGame(number_of_rounds=rounds, dims=dims, nrange=ranges)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2])) 
    
    if choice == 4:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        dims = int(input("Dims : "))
        
        max_deg = int(input("Maximum degree : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "dim" : dims, "deg" : max_deg}
        stats = gr.general_runner(gh.polyDet, rounds, inpt_dict, md)#matrixgames.polyDetGame(number_of_rounds=rounds, dims=dims, nrange=ranges, max_deg=max_deg)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    
    if choice == 5:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of abs of coeffs (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        c, d = input("Range of abs of inputs (seperated by blank space): ").split(" ")
        inp_ranges = [int(c), int(d)]
        
        deg = int(input("Polynomial degree : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "inp_ranges" : inp_ranges, "deg" : deg}
        stats = gr.general_runner(gh.polyEval, rounds, inpt_dict, md)#multgame.polyEval(rounds, deg, ranges[:], inp_ranges[:])
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 6:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of surds (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        c, d = input("Range of roots: ").split(" ")
        root_ranges = [int(c), int(d)]
        ndigits = int(input("digits after floating point : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "root_ranges" : root_ranges, "ndigits" : ndigits}
        stats = gr.general_runner(gh.evalRoot, rounds, inpt_dict, md)#multgame.evalRoot(number_of_rounds=rounds, root_range=root_ranges, ranges=ranges, ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 7:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of surds (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        c, d = input("Range of roots: ").split(" ")
        root_ranges = [int(c), int(d)]
        ndigits = int(input("digits after floating point : "))
        deg = int(input("Degree : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "root_ranges" : root_ranges, "ndigits" : ndigits, "deg" : deg}
        stats = gr.general_runner(gh.evalRootPoly, rounds, inpt_dict, md)#multgame.evalRootPoly(deg, coeffs_range=coeff_ranges, number_of_rounds=rounds, root_range=root_ranges, ranges=ranges, ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 8:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of surds (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        c, d = input("Range of roots: ").split(" ")
        root_ranges = [int(c), int(d)]
        ndigits = int(input("digits after floating point : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "root_ranges" : root_ranges, "ndigits" : ndigits}
        stats = gr.general_runner(gh.surd, rounds, inpt_dict, md)#multgame.surdGame(number_of_rounds=rounds, root_range=root_ranges, ranges=ranges, ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 9:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        a_fl = int(input("Digits after floating point : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "ndigits" : a_fl}
        stats = gr.general_runner(gh.div, rounds, inpt_dict, md)#multgame.divGame(number_of_rounds=rounds, ranges=ranges, ndigits=a_fl)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 10:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        max_deg = int(input("Maximum degree : "))
        ndigits = int(input("Digits after decimal point : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : max_deg, "ndigits" : ndigits}
        stats = gr.general_runner(gh.polyDiv, rounds, inpt_dict, md)#multgame.polyDivGame(number_of_rounds=rounds, max_deg=max_deg, nrange=ranges[:], ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 11:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        dims = int(input("Dim : "))
        ndigits = int(input("Digits after decimal point : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "dim" : dims, "ndigits" : ndigits}
        print("Enter the smallest real part of all eigen values for each matrix.")
        stats = gr.general_runner(gh.eigenValue, rounds, inpt_dict, md)#matrixgames.eigenvalueGame(number_of_rounds=rounds, dims=dims, nrange=ranges, ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 12:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of abs of roots (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        deg = int(input("Polynomial degree : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : deg}
        stats = gr.general_runner(gh.polyroots, rounds, inpt_dict, md)#multgame.polyroots(number_of_rounds=rounds, root_range=ranges[:], deg=deg)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 13:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        deg = int(input("Polynomial degree : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : deg}
        stats = gr.general_runner(gh.polydisc, rounds, inpt_dict, md)#multgame.polydisc(number_of_rounds=rounds, coeff_range=ranges[:], deg=deg)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 14:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        max_deg = int(input("Maximum degree : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : max_deg}
        stats = gr.general_runner(gh.partialFraction, rounds, inpt_dict, md)#multgame.partialFractionGame(number_of_rounds=rounds, max_deg=max_deg, nrange=ranges[:])
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 15:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        c, d = input("Range of bounds of integration (seperated by blank space): ").split(" ")
        branges = [int(c), int(d)]
        
        max_deg = int(input("Maximum degree : "))
        ndigits = int(input("Digits after floating point : "))
        mode = int(input("Enter the mode : \n 1-Rational Expressions\n 2-Algebraic Expression\n 3-Trig Expression\n 4-Shuffle\n"))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "boundary_ranges" : branges, "deg" : max_deg, "ndigits" : ndigits, "mode" : mode}
        stats = gr.general_runner(gh.subIntGame, rounds, inpt_dict, md)#multgame.integralGame(number_of_rounds=rounds, deg=max_deg, mode=mode, nranges=ranges[:], boundranges=branges[:], ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 16:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a = int(input("Number of Digits : "))
        
        os.system("clear")
        inpt_dict = {"ndigits" : a}
        stats = gr.general_runner(gh.regMulDig, rounds, inpt_dict, md)#multgame.regMulGameDig(number_of_rounds=rounds, digits=a)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    if choice == 17:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        c, d = input("Range of half period (seperated by blank space): ").split(" ")
        p_ranges = [int(c), int(d)]
        
        max_deg = int(input("Maximum degree : "))
        moe = float(input("Margin of error : "))
        exp_cond = int(input("Exponential mode? 1 : Yes\t0 : No "))
        u_cond = int(input("Poly mode? 1 : Yes\t0 : No "))
        umvar_cond = 0
        if u_cond != 0:
            umvar_cond = int(input("Multivariate mode? 1 : Yes\t0 : No "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "period_ranges" : p_ranges, "deg" : max_deg, "moe" : moe, "exp_cond" : exp_cond, "u_cond" : u_cond, "umvar_cond" : umvar_cond}
        stats = gr.general_runner(gh.fourierSeries, rounds, inpt_dict, md)#multgame.fourierSgame(number_of_rounds=rounds, nranges=ranges[:], deg=max_deg, p_range=p_ranges, exp_cond=exp_cond, u_cond=u_cond, umvar_cond=umvar_cond, moe=moe)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 18:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of coeffs (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        c, d = input("Range of answers (seperated by blank space): ").split(" ")
        param_ranges = [int(c), int(d)]
        parameters = int(input("Number of unknowns : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "params" : parameters, "param_ranges" : param_ranges}
        stats = gr.general_runner(gh.lineq, rounds, inpt_dict, md)#multgame.linearEqSystem(number_of_rounds=rounds, coeff_abs_ranges=ranges[:], parameters=parameters, param_abs_ranges=param_ranges[:])
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2])) 
    
    if choice == 19:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        max_deg = int(input("N : "))
        ndigits = int(input("Digits after floating point : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "n" : max_deg, "ndigits" : ndigits}
        stats = gr.general_runner(gh.mean, rounds, inpt_dict, md)#multgame.meanGame(number_of_rounds=rounds, nrange=ranges, n=max_deg, ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    if choice == 20:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        max_deg = int(input("N : "))
        ndigits = int(input("Digits after floating point : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "ndigits" : ndigits, "n" : max_deg}
        stats = gr.general_runner(gh.stdev, rounds, inpt_dict, md)#multgame.stdevGame(number_of_rounds=rounds, nrange=ranges, n=max_deg, ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 21:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        max_deg = int(input("Maximum degree : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : max_deg}
        stats = gr.general_runner(gh.diffeq, rounds, inpt_dict, md)#multgame.diffeq(number_of_rounds=rounds, nranges=ranges[:], max_deg=max_deg)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    
    if choice == 22:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        max_deg = int(input("Maximum degree : "))
        ndigits = int(input("Digits after floating point : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : max_deg, "ndigits" : ndigits}
        stats = gr.general_runner(gh.pcurve, rounds, inpt_dict, md)#multgame.pcurveGameC(number_of_rounds=rounds, max_deg=max_deg, nranges=ranges[:], ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 23:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        max_deg = int(input("Maximum degree : "))
        ndigits = int(input("Digits after floating point : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : max_deg, "ndigits" : ndigits}
        stats = gr.general_runner(gh.pcurveT, rounds, inpt_dict, md)#multgame.pcurveGameT(number_of_rounds=rounds, max_deg=max_deg, nranges=ranges[:], ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 24:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        max_deg = int(input("Maximum degree : "))
        moe = float(input("Margin of error : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : max_deg, "moe" : moe}
        stats = gr.general_runner(gh.lineIntegral, rounds, inpt_dict, md)#multgame.lineIntegral(number_of_rounds=rounds, max_deg=max_deg, nranges=ranges[:], moe=moe)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 25:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        max_deg = int(input("Maximum degree : "))
        ndigits = int(input("Digits after floating point : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : max_deg, "ndigits" : ndigits}
        stats = gr.general_runner(gh.divergence, rounds, inpt_dict, md)#multgame.diverganceGame(number_of_rounds=rounds, max_deg=max_deg, nranges=ranges[:], ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 26:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        max_deg = int(input("Maximum degree : "))
        moe = float(input("Margin of error : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : max_deg, "moe" : moe}
        stats = gr.general_runner(gh.lineIntegralScalar, rounds, inpt_dict, md)#multgame.lineIntegralScalar(number_of_rounds=rounds, max_deg=max_deg, nranges=ranges[:], moe=moe)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 27:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        c, d = input("Range of answers (seperated by blank space): ").split(" ")
        param_ranges = [int(c), int(d)]
        parameters = int(input("Number of unknowns : "))
        float_mode = int(input("Float mode ? (1 yes/0 no)"))
        
        max_deg = int(input("Maximum degree : "))
        ndigits = int(input("Digits after floating point : "))
        moe = float(input("Margin of error : "))
        dims = int(input("Dim : "))
        c, d = input("Range of abs of inputs (seperated by blank space): ").split(" ")
        inp_ranges = [int(c), int(d)]
        n = int(input("N : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : max_deg, "moe" : moe, "param_ranges" : param_ranges, "params":parameters, "float_mode" : float_mode, "ndigits":ndigits, "dim" : dims, "inp_ranges": inp_ranges, "n" : n, "root_ranges" : ranges}
        stats = gr.general_runner(gh.shuffle, rounds, inpt_dict, md)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 28:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        c, d = input("Range of bounds of integration (seperated by blank space): ").split(" ")
        branges = [int(c), int(d)]
        c, d = input("Range of half period (seperated by blank space): ").split(" ")
        p_ranges = [int(c), int(d)]
        inpt_dict = {"nranges" : ranges, 
                     "deg" : int(input("Maximum degree : ")),
                     "ndigits" : int(input("Digits after floating point : ")), 
                     "moe" : float(input("Margin of error : ")), 
                     "mode" : int(input("Enter the mode : \n 1-Rational Expressions\n 2-Algebraic Expression\n 3-Trig Expression\n 4-Shuffle\n")),
                     "boundary_ranges" : branges,
                     "exp_cond" : int(input("Exponential mode :\n 0-No\n 1-Yes\n")),
                     "u_cond" : int(input("Polynomial mode :\n 0-No\n 1-Yes\n")),
                     "period_ranges" : p_ranges}
        
        inpt_dict.update({"umvar_cond" : int(input("Multivariate polynomial mode :\n0-No\n1-Yes\n ")) if inpt_dict["u_cond"] else 0 })
        stats = gr.general_runner(gh.calc_suite, rounds, inpt_dict, md)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 29:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        rounds = int(input("Number of rounds : ")) if md == 1 else int(input("Duration : "))
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        c, d = input("Range of roots: ").split(" ")
        root_ranges = [int(c), int(d)]
        inpt_dict = {"nranges" : ranges,
                     "ndigits" :  int(input("Digits after floating point : ")),
                     "root_ranges" : root_ranges,
                     "dim" : int(input("Dims : ")),
                     "float_mode" : 0}
        stats = gr.general_runner(gh.arithmetic_suite, rounds, inpt_dict, md)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    
while True:
    static()
    z = input("Press Enter to continue...")
