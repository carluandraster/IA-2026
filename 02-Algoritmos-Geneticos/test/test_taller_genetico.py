"""
Suite de Evaluacion Automatizada para Taller de Algoritmos Geneticos (IA-2026)
-------------------------------------------------------------------------------
Este archivo es ejecutado automaticamente por GitHub Actions ante cada Pull Request
y tambien puede ser ejecutado localmente por los estudiantes con:
    py test_taller_genetico.py
o bien con:
    py -m pytest test_taller_genetico.py

Valida los contratos de entrada/salida y las invariantes matematicas de:
- TODO 1: Decodificacion binaria y funcion de fitness (Parabola)
- TODO 2: Seleccion por ruleta proporcional y cruce en un punto
- TODO 3: Mutacion puntual y reemplazo generacional con elitismo
- TODO 4: Fitness de Smart Rockets (cercania, velocidad y colisiones)
- TODO 5: Fitness del TSP (inversa de distancia ciclica de reparto)
- TODO 6: Fitness Criptografico (frecuencias del espanol y diccionario)
"""

import sys
import os
import json
import unittest
import numpy as np

DEFAULT_NOTEBOOK = "Taller_Algoritmos_Geneticos.ipynb"

def resolver_ruta_cuaderno(notebook_path):
    """
    Resuelve la ruta del cuaderno admitiendo ejecucion desde la raiz del repo,
    desde 02-Algoritmos-Geneticos/ o desde la subcarpeta test/.
    """
    if os.path.exists(notebook_path):
        return os.path.abspath(notebook_path)

    dir_script = os.path.dirname(os.path.abspath(__file__))
    dir_padre = os.path.dirname(dir_script)
    nombre = os.path.basename(notebook_path)

    candidatos = [
        os.path.join(dir_padre, nombre),
        os.path.join(dir_script, nombre),
        os.path.join(os.getcwd(), nombre),
        os.path.join(os.getcwd(), "02-Algoritmos-Geneticos", nombre),
        os.path.join(os.path.dirname(dir_padre), nombre),
        os.path.join(os.path.dirname(os.path.dirname(dir_padre)), nombre),
    ]
    for c in candidatos:
        if os.path.exists(c):
            return os.path.abspath(c)

    return notebook_path

def cargar_namespace_desde_notebook(notebook_path):
    """
    Lee un archivo .ipynb y ejecuta sus celdas de codigo en un namespace aislado.
    Omite directivas magicas (%pip, %matplotlib, etc.) y llamadas a plt.show().
    """
    notebook_path = resolver_ruta_cuaderno(notebook_path)
    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"No se encontro el cuaderno en la ruta: {notebook_path}")

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    namespace = {
        "__name__": "__main__",
        "np": np,
        "numpy": np
    }

    for cell in nb_data.get("cells", []):
        if cell.get("cell_type") == "code":
            lineas = cell.get("source", [])
            lineas_filtradas = []
            for linea in lineas:
                linea_strip = linea.strip()
                # Omitir comandos magicos de IPython / Colab
                if linea_strip.startswith(("%", "!", "get_ipython")):
                    continue
                # Neutralizar visualizaciones graficas y animaciones para agilizar ejecucion headless
                if "plt.show(" in linea_strip or "display(HTML(" in linea_strip:
                    lineas_filtradas.append("pass\n")
                    continue
                lineas_filtradas.append(linea)
            
            codigo_celda = "".join(lineas_filtradas)
            if codigo_celda.strip():
                try:
                    exec(codigo_celda, namespace)
                except Exception:
                    # Permite que los asserts internos del cuaderno vacio no aborten la carga
                    pass

    return namespace


# Variable global para almacenar el namespace del notebook bajo test
NOTEBOOK_NAMESPACE = None
TARGET_NOTEBOOK_PATH = None


def setup_notebook(notebook_path=None):
    global NOTEBOOK_NAMESPACE, TARGET_NOTEBOOK_PATH
    if notebook_path is None:
        if len(sys.argv) > 1 and sys.argv[1].endswith(".ipynb"):
            notebook_path = sys.argv.pop(1)
        else:
            notebook_path = DEFAULT_NOTEBOOK

    notebook_path = resolver_ruta_cuaderno(notebook_path)
    TARGET_NOTEBOOK_PATH = notebook_path
    NOTEBOOK_NAMESPACE = cargar_namespace_desde_notebook(notebook_path)


class TestTODO1DecodificacionYFitness(unittest.TestCase):
    """Pruebas para el Bloque TODO 1: Representacion y Fitness de la Parabola"""

    @classmethod
    def setUpClass(cls):
        if NOTEBOOK_NAMESPACE is None:
            setup_notebook()
        cls.ns = NOTEBOOK_NAMESPACE

    def test_01_funcion_decodificar_existe(self):
        self.assertIn("decodificar_cromosoma", self.ns,
                      "[TODO 1] La funcion 'decodificar_cromosoma' no esta definida en el cuaderno.")

    def test_02_decodificacion_valores_extremos_y_patrones(self):
        decodificar = self.ns["decodificar_cromosoma"]
        
        casos = [
            (np.array([0, 0, 0, 0, 0]), 0),
            (np.array([1, 1, 1, 1, 1]), 31),
            (np.array([0, 1, 1, 1, 1]), 15),
            (np.array([1, 0, 0, 0, 0]), 16),
            (np.array([0, 0, 0, 0, 1]), 1),
            (np.array([0, 0, 0, 1, 0]), 2),
            (np.array([0, 0, 1, 0, 0]), 4),
            (np.array([0, 1, 0, 0, 0]), 8),
            (np.array([1, 0, 1, 0, 1]), 21)
        ]
        for cromo, valor_esperado in casos:
            res = decodificar(cromo)
            self.assertIsNotNone(res, "[TODO 1] 'decodificar_cromosoma' retorno None. Completa la funcion.")
            self.assertEqual(int(res), valor_esperado,
                             f"[TODO 1] Para el cromosoma {list(cromo)} se esperaba x={valor_esperado}, pero retorno {res}.")

    def test_03_funcion_calcular_fitness_existe(self):
        self.assertIn("calcular_fitness", self.ns,
                      "[TODO 1] La funcion 'calcular_fitness' no esta definida en el cuaderno.")

    def test_04_calcular_fitness_valores_matematicos(self):
        calcular_fitness = self.ns["calcular_fitness"]

        # Vertice maximo en x=15: f(15) = 300 - (15-15)^2 = 300
        f_max = calcular_fitness(15)
        self.assertIsNotNone(f_max, "[TODO 1] 'calcular_fitness' retorno None.")
        self.assertEqual(f_max, 300, f"[TODO 1] En el optimo x=15 se esperaba fitness 300, pero dio {f_max}.")

        # Limite inferior x=0: f(0) = 300 - (-15)^2 = 75
        self.assertEqual(calcular_fitness(0), 75,
                         f"[TODO 1] Para x=0 se esperaba fitness 75, pero dio {calcular_fitness(0)}.")

        # Simetria en x=30: f(30) = 300 - (15)^2 = 75
        self.assertEqual(calcular_fitness(30), 75,
                         f"[TODO 1] Para x=30 se esperaba fitness 75, pero dio {calcular_fitness(30)}.")

        # Limite superior x=31: f(31) = 300 - (16)^2 = 44
        self.assertEqual(calcular_fitness(31), 44,
                         f"[TODO 1] Para x=31 se esperaba fitness 44, pero dio {calcular_fitness(31)}.")

        # Monotonia: x=14 debe tener mayor fitness que x=10
        self.assertGreater(calcular_fitness(14), calcular_fitness(10),
                           "[TODO 1] El fitness en x=14 debe ser mayor que en x=10.")


class TestTODO2SeleccionYCruce(unittest.TestCase):
    """Pruebas para el Bloque TODO 2: Seleccion por Ruleta y Cruzamiento en Un Punto"""

    @classmethod
    def setUpClass(cls):
        if NOTEBOOK_NAMESPACE is None:
            setup_notebook()
        cls.ns = NOTEBOOK_NAMESPACE

    def test_01_funciones_existen(self):
        self.assertIn("seleccion_ruleta", self.ns,
                      "[TODO 2] La funcion 'seleccion_ruleta' no esta definida.")
        self.assertIn("crossover_un_punto", self.ns,
                      "[TODO 2] La funcion 'crossover_un_punto' no esta definida.")

    def test_02_seleccion_ruleta_propiedades(self):
        seleccion_ruleta = self.ns["seleccion_ruleta"]
        
        poblacion = np.array([
            [0, 0, 0, 0, 0], # x=0,  fit=75
            [0, 1, 1, 1, 1], # x=15, fit=300 (muy fit)
            [1, 1, 1, 1, 1], # x=31, fit=44
        ])
        fitness = np.array([75, 300, 44])

        padres = seleccion_ruleta(poblacion, fitness, n_padres=4)
        self.assertIsNotNone(padres, "[TODO 2] 'seleccion_ruleta' retorno None.")
        self.assertEqual(len(padres), 4, f"[TODO 2] Se solicitaron 4 padres, pero se recibieron {len(padres)}.")
        
        for p in padres:
            coincide = any(np.array_equal(p, ind) for ind in poblacion)
            self.assertTrue(coincide, "[TODO 2] Un padre seleccionado no pertenecia a la poblacion original.")

    def test_03_crossover_un_punto_intercambio(self):
        crossover = self.ns["crossover_un_punto"]

        p1 = np.array([0, 0, 0, 0, 0])
        p2 = np.array([1, 1, 1, 1, 1])

        hijo1, hijo2 = crossover(p1, p2, punto_corte=2)
        self.assertIsNotNone(hijo1, "[TODO 2] hijo1 es None.")
        self.assertIsNotNone(hijo2, "[TODO 2] hijo2 es None.")

        esperado_h1 = np.array([0, 0, 1, 1, 1])
        esperado_h2 = np.array([1, 1, 0, 0, 0])

        np.testing.assert_array_equal(hijo1, esperado_h1,
                                     f"[TODO 2] hijo1 incorrecto en cruce k=2. Se esperaba {esperado_h1}, dio {hijo1}")
        np.testing.assert_array_equal(hijo2, esperado_h2,
                                     f"[TODO 2] hijo2 incorrecto en cruce k=2. Se esperaba {esperado_h2}, dio {hijo2}")


class TestTODO3MutacionYElitismo(unittest.TestCase):
    """Pruebas para el Bloque TODO 3: Mutacion y Ciclo Generacional con Elitismo"""

    @classmethod
    def setUpClass(cls):
        if NOTEBOOK_NAMESPACE is None:
            setup_notebook()
        cls.ns = NOTEBOOK_NAMESPACE

    def test_01_funciones_existen(self):
        self.assertIn("mutacion_puntual", self.ns,
                      "[TODO 3] La funcion 'mutacion_puntual' no esta definida.")
        self.assertIn("nueva_generacion", self.ns,
                      "[TODO 3] La funcion 'nueva_generacion' no esta definida.")

    def test_02_mutacion_probabilidades_extremas(self):
        mutar = self.ns["mutacion_puntual"]
        cromosoma = np.array([1, 0, 1, 0, 1])

        res_no_mut = mutar(cromosoma.copy(), prob_mutacion=0.0)
        np.testing.assert_array_equal(res_no_mut, cromosoma,
                                     "[TODO 3] Con prob_mutacion=0.0 el cromosoma no debe cambiar.")

        res_full_mut = mutar(cromosoma.copy(), prob_mutacion=1.0)
        np.testing.assert_array_equal(res_full_mut, 1 - cromosoma,
                                     "[TODO 3] Con prob_mutacion=1.0 deben invertirse todos los bits (1->0, 0->1).")

    def test_03_nueva_generacion_conservacion_tamano_y_elitismo(self):
        nueva_gen = self.ns["nueva_generacion"]

        poblacion = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ])
        fitness = np.array([75, 131, 300, 299, 299, 44])

        gen_sig = nueva_gen(poblacion, fitness, prob_mutacion=0.05, elite_count=2)
        self.assertIsNotNone(gen_sig, "[TODO 3] 'nueva_generacion' retorno None.")
        self.assertEqual(len(gen_sig), len(poblacion),
                         f"[TODO 3] La nueva generacion debe conservar el tamano de poblacion ({len(poblacion)}), pero dio {len(gen_sig)}.")

        encontrado_optimo = any(np.array_equal(ind, [0, 1, 1, 1, 1]) for ind in gen_sig[:2])
        self.assertTrue(encontrado_optimo,
                        "[TODO 3] El individuo con maximo fitness no fue conservado en la elite de la nueva generacion.")


class TestTODO4SmartRocketsFitness(unittest.TestCase):
    """Pruebas para el Bloque TODO 4: Fitness de Smart Rockets"""

    @classmethod
    def setUpClass(cls):
        if NOTEBOOK_NAMESPACE is None:
            setup_notebook()
        cls.ns = NOTEBOOK_NAMESPACE

    def test_01_funcion_existe(self):
        self.assertIn("calcular_fitness_rocket", self.ns,
                      "[TODO 4] La funcion 'calcular_fitness_rocket' no esta definida en el cuaderno.")

    def test_02_propiedades_fitness_smart_rockets(self):
        calc_fit = self.ns["calcular_fitness_rocket"]

        f_cerca = calc_fit(dist_final=3.0, reached=False, crashed=False, frames_usados=75, lifespan=75)
        self.assertIsNotNone(f_cerca, "[TODO 4] 'calcular_fitness_rocket' retorno None. Completa la funcion.")
        self.assertGreater(f_cerca, 0.0, "[TODO 4] El fitness debe ser un valor positivo.")

        f_lejos = calc_fit(dist_final=50.0, reached=False, crashed=False, frames_usados=75, lifespan=75)
        self.assertGreater(f_cerca, f_lejos, "[TODO 4] A menor distancia final al objetivo, mayor debe ser el fitness.")

        # Penalizacion por choque
        f_choco = calc_fit(dist_final=3.0, reached=False, crashed=True, frames_usados=40, lifespan=75)
        self.assertGreater(f_cerca, f_choco, "[TODO 4] Un cohete que choca debe recibir menor fitness que uno intacto a igual distancia.")

        # Bonificacion por velocidad
        f_rapido = calc_fit(dist_final=0.0, reached=True, crashed=False, frames_usados=20, lifespan=75, config="velocidad")
        f_lento  = calc_fit(dist_final=0.0, reached=True, crashed=False, frames_usados=70, lifespan=75, config="velocidad")
        self.assertGreater(f_rapido, f_lento, "[TODO 4] En config='velocidad', llegar en menos frames debe dar mayor fitness.")

        # Invarianza del tiempo en solo_cercania
        f_sc1 = calc_fit(dist_final=0.0, reached=True, crashed=False, frames_usados=20, lifespan=75, config="solo_cercania")
        f_sc2 = calc_fit(dist_final=0.0, reached=True, crashed=False, frames_usados=70, lifespan=75, config="solo_cercania")
        self.assertAlmostEqual(f_sc1, f_sc2, places=4,
                               msg="[TODO 4] En config='solo_cercania', los frames empleados no deben alterar el fitness.")


class TestTODO5TSPFitness(unittest.TestCase):
    """Pruebas para el Bloque TODO 5: Fitness del TSP en Mar del Plata"""

    @classmethod
    def setUpClass(cls):
        if NOTEBOOK_NAMESPACE is None:
            setup_notebook()
        cls.ns = NOTEBOOK_NAMESPACE

    def test_01_funcion_existe(self):
        self.assertIn("fitness_tsp", self.ns,
                      "[TODO 5] La funcion 'fitness_tsp' no esta definida en el cuaderno.")

    def test_02_propiedades_fitness_tsp(self):
        fitness_tsp = self.ns["fitness_tsp"]
        num_ciudades = self.ns.get("NUM_CIUDADES", 12)
        matriz_dists = self.ns.get("MATRIZ_DISTANCIAS", None)

        ruta_ordenada = np.arange(num_ciudades)
        fit_val = fitness_tsp(None, ruta_ordenada, 0)

        self.assertIsNotNone(fit_val, "[TODO 5] 'fitness_tsp' retorno None. Completa la funcion.")
        self.assertGreater(fit_val, 0.0, "[TODO 5] El fitness debe ser un escalar estrictamente positivo.")

        if matriz_dists is not None:
            # Crear una ruta deliberadamente mas larga (zig-zag saltando indices pares e impares)
            ruta_zigzag = np.array([i for i in range(0, num_ciudades, 2)] + [i for i in range(1, num_ciudades, 2)])
            
            d_ord = sum(matriz_dists[ruta_ordenada[i], ruta_ordenada[(i+1)%num_ciudades]] for i in range(num_ciudades))
            d_zig = sum(matriz_dists[ruta_zigzag[i], ruta_zigzag[(i+1)%num_ciudades]] for i in range(num_ciudades))
            
            fit_ord = fitness_tsp(None, ruta_ordenada, 0)
            fit_zig = fitness_tsp(None, ruta_zigzag, 1)
            
            if d_ord < d_zig:
                self.assertGreater(fit_ord, fit_zig,
                                   "[TODO 5] El recorrido de menor distancia debe tener mayor fitness (inversa de la distancia).")
            elif d_zig < d_ord:
                self.assertGreater(fit_zig, fit_ord,
                                   "[TODO 5] El recorrido de menor distancia debe tener mayor fitness (inversa de la distancia).")


class TestTODO6CifradoFitness(unittest.TestCase):
    """Pruebas para el Bloque TODO 6: Fitness Criptografico (Frecuencias y Diccionario) [BONUS TRACK]"""

    @classmethod
    def setUpClass(cls):
        if NOTEBOOK_NAMESPACE is None:
            setup_notebook()
        cls.ns = NOTEBOOK_NAMESPACE

    def test_01_funciones_existen(self):
        if "fitness_frecuencias" not in self.ns or "fitness_diccionario" not in self.ns:
            self.skipTest("[TODO 6 Bonus Track] Funciones de Fase 4 no presentes en el cuaderno.")
        self.assertIn("fitness_frecuencias", self.ns)
        self.assertIn("fitness_diccionario", self.ns)

    def test_02_fitness_frecuencias_y_diccionario(self):
        fit_freq = self.ns.get("fitness_frecuencias")
        fit_dicc = self.ns.get("fitness_diccionario")
        if fit_freq is None or fit_dicc is None:
            self.skipTest("[TODO 6 Bonus Track] Funciones de Fase 4 no encontradas.")

        clave_real = self.ns.get("CLAVE_SECRETA_ENEMIGA", self.ns.get("CLAVE_REAL", np.arange(26)))

        ff_real = fit_freq(None, clave_real, 0)
        fd_real = fit_dicc(None, clave_real, 0)

        # Si el estudiante todavia no completo el bonus track (retorna None)
        if ff_real is None and fd_real is None:
            self.skipTest("[TODO 6 Bonus Track] Funciones retornan None (Fase 4 opcional no implementada aun).")

        self.assertIsNotNone(ff_real, "[TODO 6] 'fitness_frecuencias' retorno None. Completa la funcion.")
        self.assertIsNotNone(fd_real, "[TODO 6] 'fitness_diccionario' retorno None. Completa la funcion.")

        clave_rotada = (clave_real + 13) % 26

        # Frecuencias
        ff_rotada = fit_freq(None, clave_rotada, 1)
        self.assertGreater(ff_real, 0.0, "[TODO 6] El fitness de frecuencias debe ser positivo.")
        self.assertGreater(ff_real, ff_rotada,
                           "[TODO 6] La clave real debe obtener mayor fitness de frecuencias que una clave rotada arbitraria.")

        # Diccionario
        fd_rotada = fit_dicc(None, clave_rotada, 1)
        self.assertGreaterEqual(fd_real, 100.0,
                                f"[TODO 6] La clave real debe reconocer el vocabulario del diccionario, obtuvo {fd_real}.")
        self.assertGreater(fd_real, fd_rotada,
                           "[TODO 6] La clave real debe reconocer mas palabras/letras que una clave rotada.")


def imprimir_reporte_feedback(resultados_por_fase):
    """
    Genera un reporte resumido en consola detallando el puntaje base (Fases 1, 2, 3: 2.0 pts)
    y el bonus track (Fase 4: +0.5 pts).
    """
    print("\n" + "=" * 70)
    print("   REPORTE DE EVALUACION AUTOMATIZADA: TALLER ALGORITMOS GENETICOS")
    print("=" * 70)
    print(f"Cuaderno analizado: {TARGET_NOTEBOOK_PATH}")
    print("-" * 70)

    f1_res = resultados_por_fase["Fase 1"]
    f2_res = resultados_por_fase["Fase 2"]
    f3_res = resultados_por_fase["Fase 3"]
    f4_res = resultados_por_fase["Fase 4"]

    pts_f1 = 0.8 if f1_res.wasSuccessful() else round(0.8 * max(0, (f1_res.testsRun - len(f1_res.failures) - len(f1_res.errors))) / f1_res.testsRun, 2)
    pts_f2 = 0.6 if f2_res.wasSuccessful() else round(0.6 * max(0, (f2_res.testsRun - len(f2_res.failures) - len(f2_res.errors))) / f2_res.testsRun, 2)
    pts_f3 = 0.6 if f3_res.wasSuccessful() else round(0.6 * max(0, (f3_res.testsRun - len(f3_res.failures) - len(f3_res.errors))) / f3_res.testsRun, 2)

    f4_skips = len(getattr(f4_res, "skipped", []))
    f4_completada = f4_res.wasSuccessful() and f4_skips == 0
    pts_f4 = 0.5 if f4_completada else 0.0

    puntaje_base = round(pts_f1 + pts_f2 + pts_f3, 2)
    puntaje_total = round(puntaje_base + pts_f4, 2)

    def format_status(res, skips=0):
        if skips > 0 and len(res.failures) == 0 and len(res.errors) == 0:
            return "NO COMPLETADA (Opcional)"
        return "APROBADA" if res.wasSuccessful() else "INCOMPLETA / ERRORES"

    print(f" * Fase 1 - AG Canonico (TODO 1, 2, 3)     : {pts_f1:.2f} / 0.80 pts  [{format_status(f1_res)}]")
    print(f" * Fase 2 - Smart Rockets (TODO 4)         : {pts_f2:.2f} / 0.60 pts  [{format_status(f2_res)}]")
    print(f" * Fase 3 - TSP Mar del Plata (TODO 5)     : {pts_f3:.2f} / 0.60 pts  [{format_status(f3_res)}]")
    print("-" * 70)
    print(f" PUNTAJE BASE (Fases 1, 2 y 3)             : {puntaje_base:.2f} / 2.00 pts")
    print(f" * Fase 4 - Criptoanalisis (TODO 6 Bonus)  : +{pts_f4:.2f} / +0.50 pts [{format_status(f4_res, f4_skips)}]")
    print("=" * 70)
    print(f" NOTA FINAL DEL TALLER                     : {puntaje_total:.2f} / 2.00 pts  (Max. posible 2.50 pts)")
    print("=" * 70)

    base_aprobada = f1_res.wasSuccessful() and f2_res.wasSuccessful() and f3_res.wasSuccessful()

    todas_fallas = f1_res.failures + f1_res.errors + f2_res.failures + f2_res.errors + f3_res.failures + f3_res.errors
    if len(todas_fallas) > 0:
        print("\nDETALLE DE INCONSISTENCIAS DETECTADAS EN BLOQUE OBLIGATORIO:")
        for test, msg in todas_fallas:
            nombre = test.id().split(".")[-1]
            detalle = msg.split("\n")[-2] if "\n" in msg else msg
            print(f"   * [{nombre}]: {detalle}")
        print("\nPista: Revisa los bloques correspondientes en tu cuaderno e intentalo de nuevo.")

    if len(f4_res.failures) > 0 or len(f4_res.errors) > 0:
        print("\nDETALLE DEL BONUS TRACK (FASE 4):")
        for test, msg in f4_res.failures + f4_res.errors:
            nombre = test.id().split(".")[-1]
            detalle = msg.split("\n")[-2] if "\n" in msg else msg
            print(f"   * [Bonus - {nombre}]: {detalle}")

    if base_aprobada:
        if f4_completada:
            print("\n>>> ESTADO: ¡FELICITACIONES! TALLER APROBADO CON DISTINCION (2.50 / 2.00 pts).")
        else:
            print("\n>>> ESTADO: ¡FELICITACIONES! BLOQUE OBLIGATORIO APROBADO CON EXITO (2.00 / 2.00 pts).")
            print("    (Recuerda que la Fase 4 es opcional para sumar +0.50 pts extra si deseas completarla).")
    else:
        print("\n>>> ESTADO: EL BLOQUE OBLIGATORIO NO ALCANZA EL PUNTAJE DE APROBACION.")
        print("    Debes completar y corregir las Fases 1, 2 y 3 para aprobar el taller.")
    print("=" * 70 + "\n")

    return base_aprobada


if __name__ == "__main__":
    setup_notebook()
    loader = unittest.TestLoader()

    # Fase 1
    suite_f1 = unittest.TestSuite()
    suite_f1.addTests(loader.loadTestsFromTestCase(TestTODO1DecodificacionYFitness))
    suite_f1.addTests(loader.loadTestsFromTestCase(TestTODO2SeleccionYCruce))
    suite_f1.addTests(loader.loadTestsFromTestCase(TestTODO3MutacionYElitismo))

    # Fase 2
    suite_f2 = loader.loadTestsFromTestCase(TestTODO4SmartRocketsFitness)

    # Fase 3
    suite_f3 = loader.loadTestsFromTestCase(TestTODO5TSPFitness)

    # Fase 4 (Bonus Track)
    suite_f4 = loader.loadTestsFromTestCase(TestTODO6CifradoFitness)

    runner = unittest.TextTestRunner(verbosity=1)

    print("--- Verificando Fase 1: Algoritmo Genetico Canonico (0.8 pts) ---")
    res_f1 = runner.run(suite_f1)

    print("\n--- Verificando Fase 2: Smart Rockets (0.6 pts) ---")
    res_f2 = runner.run(suite_f2)

    print("\n--- Verificando Fase 3: TSP Mar del Plata (0.6 pts) ---")
    res_f3 = runner.run(suite_f3)

    print("\n--- Verificando Fase 4: Criptoanalisis de Sustitucion (Bonus +0.5 pts) ---")
    res_f4 = runner.run(suite_f4)

    resultados = {
        "Fase 1": res_f1,
        "Fase 2": res_f2,
        "Fase 3": res_f3,
        "Fase 4": res_f4,
    }

    base_aprobada = imprimir_reporte_feedback(resultados)
    sys.exit(0 if base_aprobada else 1)

