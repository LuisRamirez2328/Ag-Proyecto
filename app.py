from flask import Flask, request, jsonify, render_template
import numpy as np
import random

app = Flask(__name__)

# Definición inicial de tareas y recursos
tareas = {}
recursos = {}

# Variables para las habilidades
habilidades_lista = ["React", "Python", "Figma", "MySQL", "Next"]

# Parámetros del algoritmo genético
TAM_POBLACION = 100
MAX_GENERACIONES = 200
TASA_MUTACION = 0.05

@app.route('/')
def index():
    return render_template('index.html', tareas=tareas, recursos=recursos, habilidades_lista=habilidades_lista)

@app.route('/agregar_tarea', methods=['POST'])
def agregar_tarea():
    nombre_tarea = request.form['nombre_tarea'].strip()
    habilidades_seleccionadas = request.form.getlist('habilidades')
    horas = request.form['horas_estimadas'].strip()
    
    if not nombre_tarea:
        return jsonify({"error": "Por favor, ingresa un nombre para la tarea."}), 400
    if not habilidades_seleccionadas:
        return jsonify({"error": "Selecciona al menos una habilidad para la tarea."}), 400
    if not horas.isdigit() or int(horas) <= 0:
        return jsonify({"error": "Especifica un número válido de horas estimadas."}), 400
    
    tareas[nombre_tarea] = {"habilidades": habilidades_seleccionadas, "horas_estimadas": int(horas)}
    return jsonify({"success": True}), 200

@app.route('/agregar_recurso', methods=['POST'])
def agregar_recurso():
    habilidades_seleccionadas = request.form.getlist('habilidades')
    
    if not habilidades_seleccionadas:
        return jsonify({"error": "Selecciona al menos una habilidad para el recurso."}), 400
    
    recurso_id = f"Persona{len(recursos)+1}"
    recursos[recurso_id] = {"habilidades": habilidades_seleccionadas, "carga": 0}
    return jsonify({"success": True}), 200

@app.route('/ejecutar_algoritmo', methods=['POST'])
def ejecutar_algoritmo():
    try:
        mejor_solucion, total_horas_asignadas = algoritmo_genetico()
        resultados = calcular_carga_trabajo(mejor_solucion, total_horas_asignadas)
        return jsonify(resultados), 200
    except Exception as e:
        print(f"Error en ejecutar_algoritmo: {str(e)}")
        return jsonify({"error": str(e)}), 500

def crear_individuo():
    individuo = np.zeros((len(recursos), len(habilidades_lista)))
    total_horas_asignadas = 0
    recurso_keys = list(recursos.keys())

    for tarea, detalles in tareas.items():
        horas_tarea = detalles['horas_estimadas']
        habilidades_tarea = [habilidades_lista.index(habilidad) for habilidad in detalles['habilidades']]
        
        recursos_validos = [i for i, recurso in enumerate(recurso_keys) 
                            if any(habilidad in recursos[recurso]['habilidades'] for habilidad in detalles['habilidades'])]
        
        if not recursos_validos:
            continue
        
        horas_por_recurso = horas_tarea / len(recursos_validos)
        for i in recursos_validos:
            habilidades_validas = [h for h in habilidades_tarea 
                                   if habilidades_lista[h] in recursos[recurso_keys[i]]['habilidades']]
            if habilidades_validas:
                horas_disponibles = min(horas_por_recurso, 40 - np.sum(individuo[i]))
                if horas_disponibles > 0:
                    horas_por_habilidad = horas_disponibles / len(habilidades_validas)
                    for h in habilidades_validas:
                        individuo[i, h] += horas_por_habilidad
        
        total_horas_asignadas += min(horas_tarea, np.sum(individuo[:, habilidades_tarea]))

    return individuo, total_horas_asignadas

def fitness(individuo, total_horas_asignadas):
    penalizacion = 0
    
    # Penalizar si no se cumplen las horas requeridas por tarea
    for tarea, detalles in tareas.items():
        horas_requeridas = detalles['horas_estimadas']
        habilidades_tarea = [habilidades_lista.index(h) for h in detalles['habilidades']]
        horas_asignadas = np.sum(individuo[:, habilidades_tarea])
        if horas_asignadas < horas_requeridas:
            penalizacion += (horas_requeridas - horas_asignadas) * 20

    # Penalizar asignaciones a habilidades que la persona no tiene
    for i, recurso in enumerate(recursos):
        habilidades = [habilidades_lista.index(skill) for skill in recursos[recurso]["habilidades"]]
        for j in range(len(habilidades_lista)):
            if j not in habilidades and individuo[i, j] > 0:
                penalizacion += individuo[i, j] * 30

    # Penalizar si algún recurso excede las 40 horas semanales
    for i in range(len(recursos)):
        horas_recurso = np.sum(individuo[i])
        if horas_recurso > 40:
            penalizacion += (horas_recurso - 40) * 25

    # Recompensar uso equilibrado de habilidades y recursos
    varianza_horas_habilidades = np.var(np.sum(individuo, axis=0))
    varianza_horas_recursos = np.var(np.sum(individuo, axis=1))
    
    # Recompensar la utilización de todos los recursos y habilidades
    recursos_utilizados = np.sum(np.sum(individuo, axis=1) > 0)
    habilidades_utilizadas = np.sum(np.sum(individuo, axis=0) > 0)
    bonus_utilizacion = (recursos_utilizados / len(recursos) + habilidades_utilizadas / len(habilidades_lista)) * 200

    return 2000 - penalizacion - varianza_horas_habilidades * 10 - varianza_horas_recursos * 10 + bonus_utilizacion

def seleccion_torneo(poblacion, tamano_torneo=3):
    seleccionados = []
    for _ in range(len(poblacion)):
        participantes = random.sample(poblacion, tamano_torneo)
        ganador = max(participantes, key=lambda x: x[1])
        seleccionados.append(ganador[0])  # Asegurar que solo el individuo sea seleccionado
    return seleccionados

def cruce(padre1, padre2):
    hijo = np.zeros_like(padre1)
    for i in range(len(recursos)):
        if random.random() < 0.5:
            hijo[i] = padre1[i]
        else:
            hijo[i] = padre2[i]
    
    # Asegurar que solo se asignen horas a habilidades válidas y no exceda 40 horas por recurso
    recurso_keys = list(recursos.keys())
    for i, recurso in enumerate(recurso_keys):
        habilidades_validas = [habilidades_lista.index(skill) for skill in recursos[recurso]["habilidades"]]
        for j in range(len(habilidades_lista)):
            if j not in habilidades_validas:
                hijo[i, j] = 0
        
        # Ajustar si excede 40 horas
        total_horas = np.sum(hijo[i])
        if total_horas > 40:
            factor = 40 / total_horas
            hijo[i] *= factor
    
    return hijo

def mutacion(individuo):
    for i in range(len(recursos)):
        if random.random() < TASA_MUTACION:
            habilidades_validas = [j for j, habilidad in enumerate(habilidades_lista) 
                                   if habilidad in recursos[list(recursos.keys())[i]]["habilidades"]]
            if habilidades_validas:
                j = random.choice(habilidades_validas)
                delta = random.uniform(-5, 5)
                nuevo_valor = max(0, individuo[i, j] + delta)
                if np.sum(individuo[i]) - individuo[i, j] + nuevo_valor <= 40:
                    individuo[i, j] = nuevo_valor
    return individuo

def algoritmo_genetico():
    poblacion = []
    while len(poblacion) < TAM_POBLACION:
        individuo, total_horas = crear_individuo()
        if np.sum(individuo) > 0:  # Asegurarse de que el individuo tenga asignaciones
            poblacion.append((individuo, total_horas))
    
    total_horas_asignadas = poblacion[0][1]
    
    for generacion in range(MAX_GENERACIONES):
        # Evaluar fitness
        poblacion = [(ind, fitness(ind, total_horas_asignadas)) for ind, _ in poblacion]
        
        # Ordenar por fitness
        poblacion.sort(key=lambda x: x[1], reverse=True)
        
        # Imprimir progreso
        if generacion % 10 == 0:
            print(f"Generación {generacion}: Mejor fitness = {poblacion[0][1]}")
        
        # Poda: mantener el 70% superior y regenerar el 30% restante
        num_mantener = int(TAM_POBLACION * 0.7)
        poblacion_podada = poblacion[:num_mantener]
        
        # Regenerar el 30% restante
        while len(poblacion_podada) < TAM_POBLACION:
            nuevo_individuo, _ = crear_individuo()
            if np.sum(nuevo_individuo) > 0:
                poblacion_podada.append((nuevo_individuo, total_horas_asignadas))
        
        # Selección
        seleccionados = seleccion_torneo(poblacion_podada)
        
        # Cruce y mutación
        nueva_poblacion = []
        while len(nueva_poblacion) < TAM_POBLACION:
            padre1, padre2 = random.sample(seleccionados, 2)
            hijo = cruce(padre1, padre2)
            hijo = mutacion(hijo)
            nueva_poblacion.append((hijo, total_horas_asignadas))
        
        # Elitismo: mantener al mejor individuo
        nueva_poblacion[0] = poblacion[0]
        
        poblacion = nueva_poblacion
    
    mejor_individuo = poblacion[0][0]
    return mejor_individuo, total_horas_asignadas

def calcular_carga_trabajo(mejor_solucion, total_horas_asignadas):
    resultados = {
        "tareas": [],
        "carga_trabajo": []
    }

    recurso_keys = list(recursos.keys())
    
    # Calcular la carga de trabajo
    for i, recurso in enumerate(recurso_keys):
        carga_recurso = {"recurso": recurso}
        total_horas_recurso = 0
        for j, habilidad in enumerate(habilidades_lista):
            horas = int(round(float(mejor_solucion[i, j])))  # Redondear al entero más cercano
            carga_recurso[habilidad] = horas
            total_horas_recurso += horas
        carga_recurso['total_horas'] = total_horas_recurso
        resultados["carga_trabajo"].append(carga_recurso)
    
    # Asignar tareas basándose en la carga de trabajo calculada
    for tarea, detalles in tareas.items():
        habilidades_requeridas = set(detalles['habilidades'])
        horas_tarea = detalles['horas_estimadas']
        recursos_asignados = []
        tiempo_real = 0

        for i, recurso in enumerate(recurso_keys):
            horas_asignadas = 0
            for habilidad in habilidades_requeridas:
                if habilidad in recursos[recurso]['habilidades']:
                    horas_disponibles = resultados["carga_trabajo"][i][habilidad]
                    if horas_disponibles > 0:
                        horas_asignadas += horas_disponibles
                        tiempo_real += horas_disponibles
            
            if horas_asignadas > 0:
                recursos_asignados.append(f"{recurso} ({horas_asignadas}h)")

        # Ajustar el margen de tiempo permitido
        margen_tiempo = int(0.1 * horas_tarea)  # 10% de margen, redondeado a entero
        estado = "Dentro del tiempo" if abs(tiempo_real - horas_tarea) <= margen_tiempo else "Fuera del tiempo"

        resultados["tareas"].append({
            "tarea": tarea,
            "recursos": ', '.join(recursos_asignados),
            "tiempo_real": tiempo_real,
            "tiempo_estimado": horas_tarea,
            "estado": estado
        })

    return resultados

if __name__ == "__main__":
    app.run(debug=True)