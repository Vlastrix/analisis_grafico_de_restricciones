"""
Aplicación Flask para Análisis Gráfico de Restricciones en Programación Lineal
Permite resolver y visualizar problemas de PL con 2 variables
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
import json

app = Flask(__name__)

def parse_constraint(constraint_str):
    """
    Parsea una restricción del formato: '2*x1 + 3*x2 <= 10'
    Retorna: coeficientes [a1, a2] y el valor b, y el tipo de restricción
    """
    try:
        # Reemplazar operadores
        constraint_str = constraint_str.replace('x1', 'x[0]').replace('x2', 'x[1]')
        constraint_str = constraint_str.replace('X1', 'x[0]').replace('X2', 'x[1]')
        
        # Identificar el tipo de restricción
        if '<=' in constraint_str:
            op = '<='
            left, right = constraint_str.split('<=')
        elif '>=' in constraint_str:
            op = '>='
            left, right = constraint_str.split('>=')
        elif '=' in constraint_str and '!' not in constraint_str:
            op = '='
            left, right = constraint_str.split('=')
        else:
            raise ValueError("Operador no válido")
        
        # Evaluar el lado derecho (valor b)
        b = float(eval(right.strip()))
        
        # Extraer coeficientes del lado izquierdo
        left = left.strip()
        
        # Inicializar coeficientes
        a1, a2 = 0, 0
        
        # Método simple: evaluar en puntos específicos
        x = [1, 0]
        val1 = eval(left)
        x = [0, 1]
        val2 = eval(left)
        x = [0, 0]
        val0 = eval(left)
        
        a1 = val1 - val0
        a2 = val2 - val0
        
        return [a1, a2], b, op
    
    except Exception as e:
        raise ValueError(f"Error al parsear restricción: {str(e)}")

def find_feasible_region(constraints, bounds):
    """
    Encuentra los vértices de la región factible
    """
    if not constraints:
        return []
    
    # Convertir restricciones a formato de semiespacios
    A_ub = []
    b_ub = []
    
    for constraint in constraints:
        coeffs, b, op = constraint
        if op == '<=':
            A_ub.append(coeffs)
            b_ub.append(b)
        elif op == '>=':
            A_ub.append([-c for c in coeffs])
            b_ub.append(-b)
    
    # Agregar restricciones de no negatividad
    A_ub.append([-1, 0])  # -x1 <= 0 => x1 >= 0
    b_ub.append(0)
    A_ub.append([0, -1])  # -x2 <= 0 => x2 >= 0
    b_ub.append(0)
    
    # Agregar límites superiores si están definidos
    if bounds['x1_max'] is not None:
        A_ub.append([1, 0])
        b_ub.append(bounds['x1_max'])
    if bounds['x2_max'] is not None:
        A_ub.append([0, 1])
        b_ub.append(bounds['x2_max'])
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Encontrar un punto interior
    try:
        # Intentar resolver un problema simple para encontrar un punto factible
        c = [0, 0]
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs', bounds=(0, None))
        
        if not result.success:
            return []
        
        interior_point = result.x
        
        # Calcular semiespacios en formato [a1, a2, -b]
        halfspaces = np.hstack([A_ub, -b_ub.reshape(-1, 1)])
        
        # Encontrar la intersección de semiespacios
        hs = HalfspaceIntersection(halfspaces, interior_point)
        vertices = hs.intersections
        
        # Ordenar vértices en sentido antihorario
        hull = ConvexHull(vertices)
        vertices = vertices[hull.vertices]
        
        return vertices.tolist()
    
    except Exception as e:
        print(f"Error al calcular región factible: {str(e)}")
        # Método alternativo: encontrar intersecciones manualmente
        return find_vertices_alternative(A_ub, b_ub)

def find_vertices_alternative(A_ub, b_ub):
    """
    Método alternativo para encontrar vértices de la región factible
    Encuentra intersecciones de pares de líneas
    """
    vertices = []
    n_constraints = len(A_ub)
    
    # Probar todas las combinaciones de 2 restricciones
    for i in range(n_constraints):
        for j in range(i+1, n_constraints):
            # Resolver sistema 2x2
            A = np.array([A_ub[i], A_ub[j]])
            b = np.array([b_ub[i], b_ub[j]])
            
            try:
                # Verificar que no sean paralelas
                det = np.linalg.det(A)
                if abs(det) < 1e-10:
                    continue
                
                # Resolver
                point = np.linalg.solve(A, b)
                
                # Verificar que el punto satisfaga todas las restricciones
                if np.all(A_ub @ point <= b_ub + 1e-6):
                    vertices.append(point.tolist())
            
            except:
                continue
    
    # Eliminar duplicados
    unique_vertices = []
    for v in vertices:
        is_duplicate = False
        for uv in unique_vertices:
            if np.allclose(v, uv, atol=1e-6):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_vertices.append(v)
    
    return unique_vertices

def solve_lp(objective_coeffs, constraints, is_maximize, bounds):
    """
    Resuelve el problema de programación lineal
    """
    if not constraints:
        return None, None
    
    # Preparar coeficientes
    c = objective_coeffs if not is_maximize else [-x for x in objective_coeffs]
    
    # Preparar restricciones
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    for constraint in constraints:
        coeffs, b, op = constraint
        if op == '<=':
            A_ub.append(coeffs)
            b_ub.append(b)
        elif op == '>=':
            A_ub.append([-x for x in coeffs])
            b_ub.append(-b)
        elif op == '=':
            A_eq.append(coeffs)
            b_eq.append(b)
    
    # Convertir a arrays numpy
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None
    
    # Resolver
    try:
        result = linprog(
            c, 
            A_ub=A_ub, 
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0, None),
            method='highs'
        )
        
        if result.success:
            optimal_value = -result.fun if is_maximize else result.fun
            return result.x.tolist(), optimal_value
        else:
            return None, None
    
    except Exception as e:
        print(f"Error al resolver LP: {str(e)}")
        return None, None

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    """Endpoint para resolver el problema de PL"""
    try:
        data = request.json
        
        # Parsear función objetivo
        obj_coeffs = [
            float(data.get('obj_x1', 0)),
            float(data.get('obj_x2', 0))
        ]
        is_maximize = data.get('objective_type', 'maximize') == 'maximize'
        
        # Parsear restricciones
        constraints = []
        constraint_strs = data.get('constraints', [])
        
        for constraint_str in constraint_strs:
            if constraint_str.strip():
                try:
                    parsed = parse_constraint(constraint_str)
                    constraints.append(parsed)
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': f"Error en restricción '{constraint_str}': {str(e)}"
                    })
        
        # Obtener límites para el gráfico
        bounds = {
            'x1_max': data.get('x1_max', 20),
            'x2_max': data.get('x2_max', 20)
        }
        
        # Encontrar región factible
        vertices = find_feasible_region(constraints, bounds)
        
        # Resolver problema de optimización
        optimal_point, optimal_value = solve_lp(obj_coeffs, constraints, is_maximize, bounds)
        
        # Preparar datos para graficar
        constraint_lines = []
        for i, (constraint_str, (coeffs, b, op)) in enumerate(zip(constraint_strs, constraints)):
            constraint_lines.append({
                'coeffs': coeffs,
                'b': b,
                'op': op,
                'label': constraint_str.strip()
            })
        
        return jsonify({
            'success': True,
            'vertices': vertices,
            'optimal_point': optimal_point,
            'optimal_value': optimal_value,
            'constraint_lines': constraint_lines,
            'objective_coeffs': obj_coeffs,
            'is_maximize': is_maximize
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
