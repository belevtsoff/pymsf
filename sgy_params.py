reference_set = dict(
    # self-coupling strength
    w = 0,

    # other parameters
    Cm = 1, # ufarad / cmeter2

    g_na = 35, # msiemens / cmeter2
    g_nap = 0.2, # msiemens / cmeter2
    g_kdr = 6, # msiemens / cmeter2
    g_kslow = 1.8, # msiemens / cmeter2
    g_l = 0.05, # msiemens / cmeter2
    g_ampa = 0.08, # msiemens / cmeter2
    g_nmda = 0.07, # msiemens / cmeter2

    V_na = 55, # mvolt
    V_k = -90, # mvolt
    V_l = -70, # mvolt
    V_glu = 0, # mvolt

    theta_m = -30, # mvolt
    theta_h = -45, # mvolt
    theta_th = -40.5, # mvolt
    theta_p = -47, # mvolt
    theta_n = -33, # mvolt
    theta_tn = -27, # mvolt
    theta_z = -39, # mvolt
    theta_s = -20, # mvolt

    sigma_m = 9.5, # mvolt
    sigma_h = -7, # mvolt
    sigma_th = -6, # mvolt
    sigma_p = 3, # mvolt
    sigma_n = 10, # mvolt
    sigma_tn = -15, # mvolt
    sigma_z = 5, # mvolt
    sigma_s = 2, # mvolt

    tau_z = 75, # msecond
    tau_ampa = 5, # msecond
    tau_hat_nmda = 14.3, # msecond
    tau_nmda = 100, # msecond

    k_fp = 1, # 1 / msecond
    k_xn = 1, # 1 / msecond
    k_fn = 1 # 1 / msecond
)

#init_conditions = list(
    #V = -70, # V, mvolt
    #h = 0.8, # h
    #n = 0.2, # n
    #z = 0.1, # z
    #s_ampa = 0.8, 
    #x_nmda = 0.0,
    #s_nmda = 0.9
#)
