--------------------------
Contenidos del archivo.py
--------------------------

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm, expon, uniform, kstest, anderson, chisquare

# --- Definimos los tests ---

    def chi_square_test(data, dist, params, bins=10):
        observed, bin_edges = np.histogram(data, bins=bins)
        expected = np.diff(dist.cdf(bin_edges, *params)) * len(data)
        expected[expected == 0] = 1e-10
        expected *= observed.sum() / expected.sum()
        chi2_stat, p_value = chisquare(observed, expected)
        return chi2_stat, p_value
    
    def ks_test(data, dist, params):
        D, p_value = kstest(data, dist.cdf, args=params)
        return D, p_value
    
    def ad_test(data, dist_name):
        if dist_name == 'norm':
            result = anderson(data, dist='norm')
        elif dist_name == 'expon':
            result = anderson(data, dist='expon')
        else:
            return None, None
        critical_value = result.critical_values[2]  # nivel 5%
        ad_stat = result.statistic
        rechaza = ad_stat > critical_value
        return ad_stat, rechaza

# --- Simulación general ---

    def run_simulation(n, n_sim=1000, alpha=0.05, bins=10, dist_null_name='norm'):
        tests = ['Chi2', 'KS', 'AD']
    
    if dist_null_name == 'norm':
        dist_null = norm
        params_null = (0, 1)
        ad_null_label = 'norm'
    elif dist_null_name == 'expon':
        dist_null = expon
        params_null = (0, 1)
        ad_null_label = 'expon'
    else:
        raise ValueError("Distribución nula no soportada para AD")
    
    if dist_null_name == 'norm':
        scenarios = {
            'Nivel (H0)': (dist_null, params_null, dist_null, params_null),
            'Alt 1: N(0.5,1)': (dist_null, params_null, norm, (0.5, 1)),
            'Alt 2: N(0,1.5)': (dist_null, params_null, norm, (0, 1.5)),
            'Alt 3: Expon(1)': (dist_null, params_null, expon, (0, 1)),
            'Alt 4: Uniform(-1,1)': (dist_null, params_null, uniform, (-1, 2))
        }
    
    if dist_null_name == 'expon':
        scenarios = {
            'Nivel (H0)': (dist_null, params_null, dist_null, params_null),
            'Alt 1: N(0,1)': (dist_null, params_null, norm, (0, 1)),
            'Alt 2: Expon(2)': (dist_null, params_null, expon, (0, 2)),
            'Alt 3: Expon(5)': (dist_null, params_null, expon, (0, 5)),
            'Alt 4: Uniform(-1,1)': (dist_null, params_null, uniform, (-1, 2))
        }
    
    results = {}

    for scenario, (dist_null_local, params_null_local, dist_alt, params_alt) in scenarios.items():
        results[scenario] = {'Chi2': {'nivel': 0, 'potencia': 0},
                             'KS': {'nivel': 0, 'potencia': 0},
                             'AD': {'nivel': 0, 'potencia': 0}}

        if scenario == 'Nivel (H0)':
            # Solo calculamos nivel
            for _ in range(n_sim):
                data_null = dist_null_local.rvs(*params_null_local, size=n)
                chi2_stat, p = chi_square_test(data_null, dist_null_local, params_null_local, bins)
                if p < alpha:
                    results[scenario]['Chi2']['nivel'] += 1
                ks_stat, p = ks_test(data_null, dist_null_local, params_null_local)
                if p < alpha:
                    results[scenario]['KS']['nivel'] += 1
                ad_stat, rechaza = ad_test(data_null, ad_null_label)
                if rechaza:
                    results[scenario]['AD']['nivel'] += 1
        else:
            # Solo calculamos potencia
            for _ in range(n_sim):
                data_alt = dist_alt.rvs(*params_alt, size=n)
                chi2_stat, p = chi_square_test(data_alt, dist_null_local, params_null_local, bins)
                if p < alpha:
                    results[scenario]['Chi2']['potencia'] += 1
                ks_stat, p = ks_test(data_alt, dist_null_local, params_null_local)
                if p < alpha:
                    results[scenario]['KS']['potencia'] += 1
                ad_stat, rechaza = ad_test(data_alt, ad_null_label)
                if rechaza:
                    results[scenario]['AD']['potencia'] += 1

    for scenario in results:
        for test in tests:
            results[scenario][test]['nivel'] /= n_sim
            results[scenario][test]['potencia'] /= n_sim

    return results

# --- Ejecutamos la simulación para varios tamaños --- NORMAL(0,1)
    n_values = [30, 100, 250]
    n_sim = 1000
    dist_null_name = 'norm'
    
    all_results = {}
    for n in n_values:
        print(f"\nSimulación para n = {n} normal") #Para ver que tamaños muestrales estamos tomando
        results = run_simulation(n, n_sim=n_sim, dist_null_name=dist_null_name)
        all_results[n] = results

# --- Graficamos resultados --- 

    for n in n_values:
        results = all_results[n]
        tests = ['Chi2', 'KS', 'AD']

    # Nivel
    nivel_scenario = 'Nivel (H0)'
    niveles = [results[nivel_scenario][test]['nivel'] for test in tests]

    plt.figure(figsize=(8, 5))
    plt.bar(tests, niveles, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylim(0, 1)
    plt.axhline(0.05, color='black', linestyle='--', label='Nivel nominal (0.05)')
    plt.title(f"Nivel empírico para n = {n}")
    plt.ylabel("Proporción de rechazos bajo H0")
    plt.legend()
    plt.show()

    # Potencia 'Alt 1: N(0.5,1)', 'Alt 2: N(0,1.5)', 'Alt 3: Expon(1)', 'Alt 4: Uniform(-1,1)'
    potencia_scenarios = ['Alt 1: N(0.5,1)', 'Alt 2: N(0,1.5)', 'Alt 3: Expon(1)', 'Alt 4: Uniform(-1,1)']
    x = np.arange(len(potencia_scenarios))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, test in enumerate(tests):
        potencias = [results[scenario][test]['potencia'] for scenario in potencia_scenarios]
        ax.bar(x + i*width, potencias, width, label=test)

    ax.set_title(f"Potencia de los tests para n = {n}")
    ax.set_ylabel("Proporción de rechazos bajo H1")
    ax.set_xticks(x + width)
    ax.set_xticklabels(potencia_scenarios, rotation=15)
    ax.legend()

    plt.tight_layout()
    plt.show()


# --- Ejecutamos la simulación para varios tamaños --- exponencial(0,1)
    n_values = [30, 100, 250]
    n_sim = 1000
    dist_null_name = 'expon'

    all_results = {}
    for n in n_values:
        print(f"\nSimulación para n = {n} exponencial") #Para ver que tamaños muestrales estamos tomando
        results = run_simulation(n, n_sim=n_sim, dist_null_name=dist_null_name)
        all_results[n] = results

# --- Graficamos resultados --- 

    for n in n_values:
        results = all_results[n]
        tests = ['Chi2', 'KS', 'AD']

    # Nivel
    nivel_scenario = 'Nivel (H0)'
    niveles = [results[nivel_scenario][test]['nivel'] for test in tests]

    plt.figure(figsize=(8, 5))
    plt.bar(tests, niveles, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylim(0, 1)
    plt.axhline(0.05, color='black', linestyle='--', label='Nivel nominal (0.05)')
    plt.title(f"Nivel empírico para n = {n}")
    plt.ylabel("Proporción de rechazos bajo H0")
    plt.legend()
    plt.show()

    # Potencia
    potencia_scenarios = ['Alt 1: N(0,1)', 'Alt 2: Expon(2)', 'Alt 3: Expon(5)', 'Alt 4: Uniform(-1,1)']
    x = np.arange(len(potencia_scenarios))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, test in enumerate(tests):
        potencias = [results[scenario][test]['potencia'] for scenario in potencia_scenarios]
        ax.bar(x + i*width, potencias, width, label=test)

    ax.set_title(f"Potencia de los tests para n = {n}")
    ax.set_ylabel("Proporción de rechazos bajo H1")
    ax.set_xticks(x + width)
    ax.set_xticklabels(potencia_scenarios, rotation=15)
    ax.legend()

    plt.tight_layout()
    plt.show()

