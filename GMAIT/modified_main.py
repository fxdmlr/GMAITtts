import os
import gamerunner as gr
import gamehandler as gh

def static(prechoice=None):
    os.system("clear")
    if prechoice is not None:
        choice = prechoice
    print("\n")
    if prechoice is None:
        choice = int(input("Enter the desired mode :\n0-Quit\n1-regMul\n2-polyMul\n3-RegDet\n4-PolyDet\n5-polyEval\n6-evalRoot\n7-evalRootPoly\n8-surdGame\n9-divGame\n10-polyDiv\n11-EigenGame\n12-RootGame\n13-DiscGame\n14-PFD\n15-IntegralGame\n16-RegDig\n17-Fourier Series\n18-Equation system\n19-Mean\n20-Stdev\n21-diffeq\n22-curvatureGame\n23-TGame\n24-LineIntegralGame\n25-DiverganceGame\n26-LineIntegralSc\n27-Shuffle\n28-FourierTransform\n29-InterpolationGame\n30-DiffeqPoly\n31-PDEConst\n32-specialPDE\n33-PDE\n34-diffeqMixed\n35-complexIntegral\n36-RealIntegral\n37-MaclaurinSeries\n38-FuncMat\n39-FuncEval\n40-RandIntegral\n41-RealIntegralHARD\n42-IntegralSolve\n43-InverseLaplace\n44-rootGameInteger\n45-NumericalAnalysis\n46-LaplaceMatrix\n47-Circuit Game\n48-DiffeqDet\n49-matrixPoly\n50-TangentProb\n51-ArithmeticGame\n52-CalcGame\n53-IntegralSet\n54-FourierSet\n55-LaplaceSet\n"))
    if choice == 1:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        dims = int(input("Dim : "))
        ndigits = int(input("Digits after decimal point : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "dim" : dims, "ndigits" : ndigits}
        print("Enter one of all real eigen values for each matrix.")
        stats = gr.general_runner(gh.eigenValue, rounds, inpt_dict, md)#matrixgames.eigenvalueGame(number_of_rounds=rounds, dims=dims, nrange=ranges, ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 12:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        c, d = input("Range of bounds of integration (seperated by blank space): ").split(" ")
        branges = [int(c), int(d)]
        
        max_deg = int(input("Maximum degree : "))
        ndigits = int(input("Digits after floating point : "))
        mode = int(input("Enter the mode : \n 1-Rational Expressions\n 2-Algebraic Expression\n 3-Trig Expression\n 4-Rational Expression(random)\n 5-Shuffle\n"))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "boundary_ranges" : branges, "deg" : max_deg, "ndigits" : ndigits, "mode" : mode}
        stats = gr.general_runner(gh.subIntGame, rounds, inpt_dict, md)#multgame.integralGame(number_of_rounds=rounds, deg=max_deg, mode=mode, nranges=ranges[:], boundranges=branges[:], ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 16:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a = int(input("Number of Digits : "))
        
        os.system("clear")
        inpt_dict = {"ndigits" : a}
        stats = gr.general_runner(gh.regMulDig, rounds, inpt_dict, md)#multgame.regMulGameDig(number_of_rounds=rounds, digits=a)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    if choice == 17:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        n_partite = int(input("Number of parts : "))
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
        inpt_dict = {"nranges" : ranges, "period_ranges" : p_ranges, "deg" : max_deg, "moe" : moe, "exp_cond" : exp_cond, "u_cond" : u_cond, "umvar_cond" : umvar_cond, "n_partite" : n_partite}
        stats = gr.general_runner(gh.fourierSeries, rounds, inpt_dict, md)#multgame.fourierSgame(number_of_rounds=rounds, nranges=ranges[:], deg=max_deg, p_range=p_ranges, exp_cond=exp_cond, u_cond=u_cond, umvar_cond=umvar_cond, moe=moe)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 18:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        order = int(input("Order : "))
        a, b = input("Range of input (seperated by blank space): ").split(" ")
        inp_ranges = [int(a), int(b)]
        
        max_deg = int(input("Maximum degree : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : max_deg, "ord" : order, 'inprange':inp_ranges[:]}
        stats = gr.general_runner(gh.diffeq, rounds, inpt_dict, md)#multgame.diffeq(number_of_rounds=rounds, nranges=ranges[:], max_deg=max_deg)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    
    if choice == 22:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
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
        inpt_dict = {"nranges" : ranges, "deg" : max_deg, "moe" : moe, "param_ranges" : param_ranges, "params":parameters, "float_mode" : float_mode, "ndigits":ndigits, "dim" : dims, "inp_ranges": inp_ranges, "n" : n}
        stats = gr.general_runner(gh.shuffle, rounds, inpt_dict, md)#multgame.lineIntegralScalar(number_of_rounds=rounds, max_deg=max_deg, nranges=ranges[:], moe=moe)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 28:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        n_partite = int(input("Number of parts : "))
        c, d = input("Range of the length of non-zero interval (seperated by blank space): ").split(" ")
        p_ranges = [int(c), int(d)]
        
        max_deg = int(input("Maximum degree : "))
        moe = float(input("Margin of error : "))
        exp_cond = int(input("Exponential mode? 1 : Yes\t0 : No "))
        u_cond = int(input("Poly mode? 1 : Yes\t0 : No "))
        umvar_cond = 0
        if u_cond != 0:
            umvar_cond = int(input("Multivariate mode? 1 : Yes\t0 : No "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "period_ranges" : p_ranges, "deg" : max_deg, "moe" : moe, "exp_cond" : exp_cond, "u_cond" : u_cond, "umvar_cond" : umvar_cond, "n_partite" : n_partite}
        stats = gr.general_runner(gh.fourierTransform, rounds, inpt_dict, md)#multgame.fourierSgame(number_of_rounds=rounds, nranges=ranges[:], deg=max_deg, p_range=p_ranges, exp_cond=exp_cond, u_cond=u_cond, umvar_cond=umvar_cond, moe=moe)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 29:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of coeffs (seperated by blank space): ").split(" ")
        ranges_coeffs = [int(a), int(b)]
        
        c, d = input("Range of inputs (seperated by blank space): ").split(" ")
        ranges_inps = [int(c), int(d)]
        
        n = int(input("N : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges_coeffs, "nranges-inps" : ranges_inps, "n" : n}
        stats = gr.general_runner(gh.interpolationGame, rounds, inpt_dict, md)#multgame.stdevGame(number_of_rounds=rounds, nrange=ranges, n=max_deg, ndigits=ndigits)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 30:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        max_deg_coeffs = int(input("Maximum degree for coeffs: "))
        max_deg_rhs = int(input("Maximum degree for rhs: "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "degc" : max_deg_coeffs, "deg" : max_deg_rhs}
        stats = gr.general_runner(gh.diffeqPoly, rounds, inpt_dict, md)#multgame.diffeq(number_of_rounds=rounds, nranges=ranges[:], max_deg=max_deg)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 31:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of coeffs (seperated by blank space): ").split(" ")
        ranges_coeffs = [int(a), int(b)]
        
        c, d = input("Range of boundaries (seperated by blank space): ").split(" ")
        ranges_inps = [int(c), int(d)]
        
        moe = float(input("Margin of error : "))
        sep = int(input("Seperablity : "))

        os.system("clear")
        inpt_dict = {"nranges" : ranges_coeffs, "l-ranges" : ranges_inps, "moe" : moe, "sep" : sep}
        stats = gr.general_runner(gh.PDEConst, rounds, inpt_dict, md)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 32:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of coeffs (seperated by blank space): ").split(" ")
        ranges_coeffs = [int(a), int(b)]
        
        c, d = input("Range of boundaries (seperated by blank space): ").split(" ")
        ranges_inps = [int(c), int(d)]
        
        moe = float(input("Margin of error : "))

        os.system("clear")
        inpt_dict = {"nranges" : ranges_coeffs, "l-ranges" : ranges_inps, "moe" : moe}
        stats = gr.general_runner(gh.PDESpecial, rounds, inpt_dict, md)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 33:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of coeffs (seperated by blank space): ").split(" ")
        ranges_coeffs = [int(a), int(b)]
        
        c, d = input("Range of boundaries (seperated by blank space): ").split(" ")
        ranges_inps = [int(c), int(d)]
        
        moe = float(input("Margin of error : "))
        sep = int(input("Seperablity : "))

        os.system("clear")
        inpt_dict = {"nranges" : ranges_coeffs, "l-ranges" : ranges_inps, "moe" : moe, "sep" : sep}
        stats = gr.general_runner(gh.PDE, rounds, inpt_dict, md)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 34:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        order = int(input("Order : "))
        
        max_deg = int(input("Maximum degree : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "deg" : max_deg, "ord":order}
        stats = gr.general_runner(gh.diffeq_mixed, rounds, inpt_dict, md)#multgame.diffeq(number_of_rounds=rounds, nranges=ranges[:], max_deg=max_deg)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 35:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        clsd = int(input("Over a closed curve? 1-Yes, 0-No : "))
        branges = [0, 0]
        if not clsd:
            a, b = input("Range of boundaries (seperated by blank space): ").split(" ")
            branges = [int(a), int(b)]
        max_deg = int(input("Maximum degree : "))
        moe = float(input("Margin of error : "))
        n = int(input("Number of parts : "))
        repeat = int(input("Repeat roots? 1-Yes 0-No "))
        mrep = 0
        if repeat:
            mrep = int(input("Maximum number of repetition : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "mdeg" : max_deg, "clsd":clsd, "n":n, "moe":moe, "rep" : repeat, "branges" : branges, "mrep" : mrep}
        stats = gr.general_runner(gh.complex_integral, rounds, inpt_dict, md)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    if choice == 36:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        mode = int(input("Mode : 0-rational 1-Trig : "))
        max_deg = 0
        if not mode:
            max_deg = int(input("Maximum degree : "))
        moe = float(input("Margin of error : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "mdeg" : max_deg, "moe" : moe, "mode" : mode}
        stats = gr.general_runner(gh.integral_cmplx, rounds, inpt_dict, md)#multgame.lineIntegralScalar(number_of_rounds=rounds, max_deg=max_deg, nranges=ranges[:], moe=moe)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 37:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        max_deg = int(input("Maximum degree : "))
        moe = float(input("Margin of error : "))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, "mdeg" : max_deg, "moe" : moe}
        stats = gr.general_runner(gh.maclaurin_series, rounds, inpt_dict, md)#multgame.lineIntegralScalar(number_of_rounds=rounds, max_deg=max_deg, nranges=ranges[:], moe=moe)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 38:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        
        input_digits = int(input("Input digits after floating point : "))
        digits_output = int(input("Result digits after floating point : "))
        dim = int(input("Dim : "))
        
        inpt_dict = {"ndigits" : input_digits, "dig" : digits_output, "dim" : dim}
        stats = gr.general_runner(gh.funcMatDet, rounds, inpt_dict, md)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 39:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        
        input_digits = int(input("Input digits after floating point : "))
        digits_output = int(input("Result digits after floating point : "))
        N = int(input("Number of functions to multiply : "))
        
        inpt_dict = {"ndigits" : input_digits, "dig" : digits_output, "N" : N}
        stats = gr.general_runner(gh.funcEval, rounds, inpt_dict, md)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 40:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        nranges = [int(a), int(b)]
        c, d = input("Range of boundaries (seperated by blank space): ").split(" ")
        branges = [int(c), int(d)]
        
        max_deg = int(input("Maximum polynomial degree : "))
        n = int(input("N : "))
        k = float(input("K : "))
        comp = int(input("Number of Comps : "))
        sums = int(input("Number of Sums : "))
        prod = int(input("Number of Prods : "))
        moe = float(input("Margin of Error : "))
        
        os.system("clear")
        inpt_dict = {"nranges" : nranges, "branges" : branges, "maxd" : max_deg, "n":n, "k":k, "moe": moe, "comp":comp, "sums":sums, "prod":prod}
        stats = gr.general_runner(gh.realIntGame, rounds, inpt_dict, md)#multgame.polyEval(rounds, deg, ranges[:], inp_ranges[:])
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 41:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        nranges = [int(a), int(b)]
        c, d = input("Range of boundaries (seperated by blank space): ").split(" ")
        branges = [int(c), int(d)]
        
        max_deg = int(input("Maximum polynomial degree : "))
        n = int(input("N : "))
        print("Enter function probability\nweights according to the table:")
        print("sin cos tan log exp sqrt asin atan poly trig_poly\n")
        fweights = input("Function Weights : ")
        print("Enter operator probability\nweights according to the table:")
        print("Addition Composition Mult. Div. ")
        wweights = input("Op. weights : ")
        moe = float(input("Margin of error: "))
        os.system("clear")
        inpt_dict = {"nranges" : nranges, "branges" : branges, "maxd" : max_deg, "n":n, "moe": moe, "fweights":fweights, "wweights":wweights}
        stats = gr.general_runner(gh.realIntGameHARD, rounds, inpt_dict, md)#multgame.polyEval(rounds, deg, ranges[:], inp_ranges[:])
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 42:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        nranges = [int(a), int(b)]
        c, d = input("Range of boundaries (seperated by blank space): ").split(" ")
        branges = [int(c), int(d)]
        moe = float(input("Margin of error: "))
        tp = int(input("Type : \n0-rational \n1-by parts \n2-rational_sqrt \n3-trig \n4-poly\n"))
        inpt_dict = {"nranges" : nranges, "branges" : branges, "moe": moe, "type":tp}
        if tp == 0:
            max_deg = int(input("Maximum polynomial degree : "))
            a, b = input("Range of numbers(n) (seperated by blank space): ").split(" ")
            n_ranges = [int(a), int(b)]
            a, b = input("Range of numbers(m) (seperated by blank space): ").split(" ")
            m_ranges = [int(a), int(b)]
            inpt_dict.update({"mdeg" : max_deg, "n_range":n_ranges, "m_range":m_ranges})
        elif tp == 1:
            max_deg = int(input("Maximum polynomial degree : "))
            n = int(input("number of functions : "))
            mdeg_c = int(input("Max composition poly degree : "))
            inpt_dict.update({"mdeg" : max_deg, "n":n, "mdeg_c":mdeg_c})
        
        elif tp == 2:
            max_deg = int(input("Maximum polynomial degree : "))
            deg_i = int(input("deg : "))
            a, b = input("Range of numbers(n) (seperated by blank space): ").split(" ")
            n_ranges = [int(a), int(b)]
            inpt_dict.update({"mdeg" : max_deg, "n_range":n_ranges, "degi":deg_i})
        
        elif tp == 3:
            pass
        elif tp == 4:
            a, b = input("Range of numbers(n) (seperated by blank space): ").split(" ")
            n_ranges = [int(a), int(b)]
            inpt_dict.update({"n_range":n_ranges})
        os.system("clear")
        stats = gr.general_runner(gh.solvableInt, rounds, inpt_dict, md)#multgame.polyEval(rounds, deg, ranges[:], inp_ranges[:])
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 43:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        nranges = [int(a), int(b)]
        c, d = input("Range of input (seperated by blank space): ").split(" ")
        branges = [int(c), int(d)]
        
        max_deg = int(input("Maximum polynomial degree : "))
        special_mode = int(input("Special functions : 1-Yes 0-No : "))
        moe = float(input("Margin of Error : "))
        os.system("clear")
        inpt_dict = {"nranges":nranges[:], "tranges":branges, "mdeg":max_deg, "moe":moe, "diffint":special_mode}
        stats = gr.general_runner(gh.inv_laplace_game, rounds, inpt_dict, md)#multgame.polyEval(rounds, deg, ranges[:], inp_ranges[:])
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    if choice == 44:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers whose nth root is to be found (seperated by blank space): ").split(" ")
        nranges = [int(a), int(b)]
        c, d = input("Range of n (seperated by blank space): ").split(" ")
        rranges = [int(c), int(d)]
        inpt_dict = {"nranges":nranges, "rrange":rranges}
        stats = gr.general_runner(gh.root_game_integer, rounds, inpt_dict, md)#multgame.polyEval(rounds, deg, ranges[:], inp_ranges[:])
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    if choice == 45:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        num_ranges = [int(a), int(b)]
        c, d = input("Range of rational numbers (seperated by blank space): ").split(" ")
        rat_ranges = [int(c), int(d)]
        
        var_num = int(input("number of variables (max 3): "))
        ppart_num = int(input("number of parts (min 1) : "))
        pure_arith = int(input("Purely arithmetic ? \n1-Yes\n0-No\n"))
        fun_ranges = [0, 1]
        inp_ndigit = 1
        res_ndigits = int(input("Digits after floating point for result : "))
        if not pure_arith:
            c, d = input("Range of function input numbers (seperated by blank space): ").split(" ")
            fun_ranges = [int(c), int(d)]
            
            inp_ndigit = int(input("Function input digits after floating point : "))
            
            
            
        inpt_dict = {"numranges" : num_ranges[:], "ratranges":rat_ranges[:], "funranges":fun_ranges[:], "ppartsnum":ppart_num, "varnumber":var_num, "purearith":pure_arith, "inpndigit":inp_ndigit, "resndigit":res_ndigits}
        stats = gr.general_runner(gh.numerical_analysis, rounds, inpt_dict, md)#multgame.polyEval(rounds, deg, ranges[:], inp_ranges[:])
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 46:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        dims = int(input("Dims : "))
        
        mdeg = int(input("Maximum degree (Matrix): "))
        mdeg_rhs = int(input("Maximum degree (R.H.S): "))
        moe = float(input("Margin of error: "))
        ndig = int(input("input digits after floating point: "))
        os.system("clear")
        print("For the Equation Ax=B Find\nthe length of inverse laplace\ntransform of x at t.\n")
        inpt_dict = {"nranges" : ranges, "dim" : dims, "mdeg" : mdeg, "mdeg_rhs":mdeg_rhs, "moe":moe, "ndigits_t":ndig}

        stats = gr.general_runner(gh.inv_lap_mat, rounds, inpt_dict, md)#matrixgames.polyDetGame(number_of_rounds=rounds, dims=dims, nrange=ranges, max_deg=max_deg)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    if choice == 47:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        
        c, d = input("Time input range (seperated by blank space): ").split(" ")
        tranges = [int(c), int(d)]
        nnode = int(input('Number of nodes : '))
        nmesh = int(input('Number of Meshs : '))
        moe = float(input('margin of error : '))
        prob = int(input('Odds of a source appearing (One in ) : '))
        os.system("clear")
        inpt_dict = {"nranges" : ranges, 'tranges':tranges, 'nnode':nnode, 'nmesh':nmesh, 'moe':moe, 'prob':prob}
        stats = gr.general_runner(gh.circuit_game, rounds, inpt_dict, md)#multgame.polyMulGame(number_of_rounds=rounds, max_deg=max_deg, nrange=ranges[:])
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 48:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        ndig = int(input('Number of digits after floating point : '))
        dim = int(input('Matrix dimentions : '))
        
        a, b = input("Range of input (seperated by blank space): ").split(" ")
        inp_ranges = [int(a), int(b)]
        
        max_deg = int(input("Maximum degree (in matrix): "))
        max_deg_l = int(input("Maximum degree: "))
        
        os.system("clear")
        inpt_dict = {"ndig" : ndig, "matdeg" : max_deg, 'mdeg' : max_deg_l, 'dim':dim, 'inprange':inp_ranges[:]}
        stats = gr.general_runner(gh.diffDet, rounds, inpt_dict, md)#multgame.diffeq(number_of_rounds=rounds, nranges=ranges[:], max_deg=max_deg)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 49:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of matrix entries (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        dim = int(input('Matrix dimentions : '))
        
        a, b = input("Range of polynomial coeffs (seperated by blank space): ").split(" ")
        inp_ranges = [int(a), int(b)]
        
        #max_deg = int(input("Maximum degree (in matrix): "))
        max_deg_l = int(input("Maximum polynomial degree: "))
        
        os.system("clear")
        inpt_dict = {"mnranges" : ranges, 'deg' : max_deg_l, 'dim':dim, 'pnranges':inp_ranges[:]}
        stats = gr.general_runner(gh.matrixPoly, rounds, inpt_dict, md)#multgame.diffeq(number_of_rounds=rounds, nranges=ranges[:], max_deg=max_deg)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 50:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a, b = input("Range of numbers (seperated by blank space): ").split(" ")
        ranges = [int(a), int(b)]
        os.system("clear")
        inpt_dict = {'nranges':ranges[:]}
        stats = gr.general_runner(gh.tangent_line, rounds, inpt_dict, md)#multgame.diffeq(number_of_rounds=rounds, nranges=ranges[:], max_deg=max_deg)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 51:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        a = int(input("Number of Digits after floating point : "))
        n = input('Number of operations (DEFAULT 3): ')
        if n == "":
            n = 3
        else:
            n = int(n)
            
        b = input('Accurate to how many digits? (DEFAULT 2) ')
        if b == "":
            b = 2
        else:
            b = int(b)
        sq = input('Include square roots ? (1-Yes 0-No) (DEFAULT NO)')
        if sq == '':
            sq = 0
        else :
            sq = int(sq)
        os.system("clear")
        inpt_dict = {"ndig" : a, 'n':n, 'ndigits':b, 'sq':sq}
        stats = gr.general_runner(gh.arithmetic_game, rounds, inpt_dict, md)#multgame.regMulGameDig(number_of_rounds=rounds, digits=a)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 52:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        moe = float(input('Margin of error : '))
        inpt_dict = {'moe' : moe}
        stats = gr.general_runner(gh.calc_game, rounds, inpt_dict, md)#multgame.regMulGameDig(number_of_rounds=rounds, digits=a)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
        
    if choice == 53:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        moe = float(input('Margin of error : '))
        inpt_dict = {'moe' : moe}
        stats = gr.general_runner(gh.integral_set_game, rounds, inpt_dict, md)#multgame.regMulGameDig(number_of_rounds=rounds, digits=a)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 54:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        moe = float(input('Margin of error : '))
        inpt_dict = {'moe' : moe}
        stats = gr.general_runner(gh.fourier_set_game, rounds, inpt_dict, md)#multgame.regMulGameDig(number_of_rounds=rounds, digits=a)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))
    
    if choice == 55:
        md = int(input("Mode :\n 1-Static\n 2-Dynamic\n"))
        roundd = int(input("Number of rounds : ")) 
        t = 0
        if md == 2:
            t = int(input("Duration : "))
        rounds = (t, roundd)
        moe = float(input('Margin of error : '))
        inpt_dict = {'moe' : moe}
        stats = gr.general_runner(gh.laplace_set_game, rounds, inpt_dict, md)#multgame.regMulGameDig(number_of_rounds=rounds, digits=a)
        print("Score : ", round(stats[0]))
        print("Total time spent : ", round(stats[1]))
        print("Time spent per item : ", round(stats[2]))

def run():        
    while True:
        static()
        z = input("Press Enter to continue ...")

if __name__ == '__main__':
    run()