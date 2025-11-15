"""
Aplicación Flask para Análisis Gráfico de Restricciones en Programación Lineal
Permite resolver y visualizar problemas de PL con 2 variables
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
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
    Encuentra los vértices de la región factible.
    Maneja regiones no acotadas añadiendo límites artificiales grandes para graficar.
    Usa método de intersecciones manuales para evitar problemas con Qhull.
    """
    if not constraints:
        return []
    
    # Convertir restricciones a formato de semiespacios
    A_ub = []
    b_ub = []
    has_x1_nonneg = False  # Bandera para x1 >= 0
    has_x2_nonneg = False  # Bandera para x2 >= 0
    
    for constraint in constraints:
        coeffs, b, op = constraint
        if op == '<=':
            A_ub.append(coeffs)
            b_ub.append(b)
        elif op == '>=':
            A_ub.append([-c for c in coeffs])
            b_ub.append(-b)
            # Detectar si ya existe x1 >= 0 o x2 >= 0
            if coeffs == [1, 0] and b == 0:
                has_x1_nonneg = True
            elif coeffs == [0, 1] and b == 0:
                has_x2_nonneg = True
    
    # Agregar restricciones de no negatividad SOLO si no fueron ingresadas por el usuario
    if not has_x1_nonneg:
        A_ub.append([-1, 0])  # -x1 <= 0 => x1 >= 0
        b_ub.append(0)
    if not has_x2_nonneg:
        A_ub.append([0, -1])  # -x2 <= 0 => x2 >= 0
        b_ub.append(0)
    
    # Agregar límites superiores si están definidos
    # Si no están definidos (región potencialmente no acotada), usar límites grandes para graficar
    x1_limit = bounds['x1_max'] if bounds['x1_max'] is not None else 1000
    x2_limit = bounds['x2_max'] if bounds['x2_max'] is not None else 1000
    
    A_ub.append([1, 0])
    b_ub.append(x1_limit)
    A_ub.append([0, 1])
    b_ub.append(x2_limit)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Usar método de intersecciones manuales (más robusto que HalfspaceIntersection)
    try:
        vertices = find_vertices_alternative(A_ub, b_ub)
        
        # Manejar casos degenerados
        if len(vertices) == 0:
            return []
        elif len(vertices) == 1:
            # Región es un solo punto
            return vertices
        elif len(vertices) == 2:
            # Región es un segmento (línea)
            return vertices
        else:
            # Región es un polígono: ordenar vértices usando ConvexHull
            try:
                vertices_array = np.array(vertices)
                hull = ConvexHull(vertices_array)
                vertices = vertices_array[hull.vertices].tolist()
            except Exception as e:
                # Si ConvexHull falla (puntos colineales), mantener vértices sin ordenar
                print(f"ConvexHull fallido: {e}")
                pass
            return vertices
    
    except Exception as e:
        print(f"Error al calcular región factible: {str(e)}")
        return []

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
    Resuelve el problema de programación lineal.
    Retorna: (optimal_point, optimal_value, status)
    status puede ser: 'optimal', 'unbounded', 'infeasible', 'error'
    """
    if not constraints:
        return None, None, 'error'
    
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
    
    # Añadir límites superiores de x1_max y x2_max como restricciones
    # Solo si están especificados y no son None
    if bounds.get('x1_max') is not None:
        A_ub.append([1, 0])  # x1 <= x1_max
        b_ub.append(bounds['x1_max'])
    if bounds.get('x2_max') is not None:
        A_ub.append([0, 1])  # x2 <= x2_max
        b_ub.append(bounds['x2_max'])
    
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
            return result.x.tolist(), optimal_value, 'optimal'
        elif result.status == 3:
            # Status 3 = unbounded
            return None, None, 'unbounded'
        elif result.status == 2:
            # Status 2 = infeasible
            return None, None, 'infeasible'
        else:
            return None, None, 'error'
    
    except Exception as e:
        print(f"Error al resolver LP: {str(e)}")
        return None, None, 'error'

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    """Endpoint para resolver el problema de PL"""
    try:
        data = request.get_json(force=True)

        # --- Parsear función objetivo (soporta dos formatos) ---
        # Formato estructurado: { "objective": { "sense": "max", "coefficients": [3,4] } }
        # Formato antiguo: campos individuales obj_x1 / obj_x2 y objective_type
        obj_coeffs = [0.0, 0.0]
        is_maximize = True
        if isinstance(data.get('objective'), dict):
            obj = data.get('objective', {})
            coeffs = obj.get('coefficients') or obj.get('coeff') or []
            if len(coeffs) >= 2:
                obj_coeffs = [float(coeffs[0]), float(coeffs[1])]
            sense = obj.get('sense') or obj.get('type')
            if sense:
                is_maximize = str(sense).lower() in ('max', 'maximize')
        else:
            # Fallback al formato por inputs individuales
            obj_coeffs = [
                float(data.get('obj_x1', 0)),
                float(data.get('obj_x2', 0))
            ]
            is_maximize = data.get('objective_type', 'maximize') == 'maximize'

        # --- Parsear restricciones (soporta cadenas y objetos) ---
        constraints = []
        constraint_inputs = data.get('constraints', [])

        for item in constraint_inputs:
            # Si llega un objeto con coeficientes y rhs
            if isinstance(item, dict):
                try:
                    coeffs = item.get('coefficients') or item.get('coeff')
                    b = item.get('rhs') if 'rhs' in item else item.get('b')
                    op = item.get('op') if 'op' in item else item.get('operator', '<=')
                    if coeffs is None or b is None:
                        return jsonify({'success': False, 'error': f"Restricción incompleta: {item}"})
                    constraints.append(([float(coeffs[0]), float(coeffs[1])], float(b), str(op)))
                except Exception as e:
                    return jsonify({'success': False, 'error': f"Error en restricción estructurada {item}: {str(e)}"})
            else:
                # Texto: "2*x1 + x2 <= 10"
                try:
                    if isinstance(item, str) and item.strip():
                        parsed = parse_constraint(item)
                        constraints.append(parsed)
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': f"Error en restricción '{item}': {str(e)}"
                    })

        # Obtener límites para el gráfico (soporta campo 'bounds' o x1_max/x2_max)
        x1_max = None
        x2_max = None
        if isinstance(data.get('bounds'), list) and len(data.get('bounds')) >= 2:
            # bounds: [[0, null], [0, null]] or [x1_max, x2_max]
            b = data.get('bounds')
            # If it's a pair of ranges, try to extract upper bounds if present
            try:
                # If it's like [[0,null],[0,null]] then no upper bounds
                if all(isinstance(x, list) for x in b):
                    # try to read second element of each bound if numeric
                    x1_upper = b[0][1]
                    x2_upper = b[1][1]
                    x1_max = None if x1_upper is None else float(x1_upper)
                    x2_max = None if x2_upper is None else float(x2_upper)
                else:
                    # simple list [x1_max, x2_max]
                    x1_max = None if b[0] is None else float(b[0])
                    x2_max = None if b[1] is None else float(b[1])
            except Exception:
                x1_max = None
                x2_max = None

        # fallback to explicit fields or defaults
        if x1_max is None:
            x1_max = data.get('x1_max', 20)
        if x2_max is None:
            x2_max = data.get('x2_max', 20)

        bounds = {
            'x1_max': None if x1_max is None else float(x1_max),
            'x2_max': None if x2_max is None else float(x2_max)
        }
        
        # Encontrar región factible
        vertices = find_feasible_region(constraints, bounds)
        
        # Filtrar restricciones redundantes de no-negatividad (x1>=0, x2>=0)
        # porque linprog ya las asume con bounds=(0,None)
        constraints_for_solver = []
        for coeffs, b, op in constraints:
            # Filtrar x1 >= 0: coeffs=[1,0], b=0, op='>='
            if coeffs == [1, 0] and b == 0 and op == '>=':
                continue  # Saltar, ya está en bounds
            # Filtrar x2 >= 0: coeffs=[0,1], b=0, op='>='
            if coeffs == [0, 1] and b == 0 and op == '>=':
                continue  # Saltar, ya está en bounds
            constraints_for_solver.append((coeffs, b, op))
        
        # Resolver problema de optimización
        optimal_point, optimal_value, status = solve_lp(obj_coeffs, constraints_for_solver, is_maximize, bounds)

        # Manejar caso de problema no acotado
        if status == 'unbounded':
            return jsonify({
                'success': False,
                'unbounded': True,
                'error': 'Problema no acotado: la región factible es infinita y la función objetivo puede crecer sin límite.' if is_maximize else 'Problema no acotado: la región factible es infinita y la función objetivo puede decrecer sin límite.',
                'vertices': vertices,
                'constraint_lines': [
                    {
                        'coeffs': coeffs,
                        'b': b,
                        'op': op,
                        'label': (orig.strip() if isinstance(orig, str) else orig.get('label') or f"{coeffs[0]}*x1 + {coeffs[1]}*x2 {op} {b}") if isinstance(orig, (str, dict)) else f"{coeffs[0]}*x1 + {coeffs[1]}*x2 {op} {b}"
                    }
                    for orig, (coeffs, b, op) in zip(constraint_inputs, constraints)
                ],
                'objective_coeffs': obj_coeffs,
                'is_maximize': is_maximize
            })
        
        # Manejar caso infactible
        if status == 'infeasible':
            return jsonify({
                'success': False,
                'infeasible': True,
                'error': 'Problema infactible: no existe ningún punto que satisfaga todas las restricciones.',
                'constraint_lines': [
                    {
                        'coeffs': coeffs,
                        'b': b,
                        'op': op,
                        'label': (orig.strip() if isinstance(orig, str) else orig.get('label') or f"{coeffs[0]}*x1 + {coeffs[1]}*x2 {op} {b}") if isinstance(orig, (str, dict)) else f"{coeffs[0]}*x1 + {coeffs[1]}*x2 {op} {b}"
                    }
                    for orig, (coeffs, b, op) in zip(constraint_inputs, constraints)
                ],
                'objective_coeffs': obj_coeffs,
                'is_maximize': is_maximize
            })

        # Identificar todos los puntos óptimos entre los vértices (si hay múltiples soluciones)
        optimal_points = []
        try:
            if vertices and optimal_value is not None:
                tol = 1e-6
                for v in vertices:
                    val = float(obj_coeffs[0]) * float(v[0]) + float(obj_coeffs[1]) * float(v[1])
                    if abs(val - optimal_value) <= tol:
                        optimal_points.append([float(v[0]), float(v[1])])
        except Exception:
            optimal_points = []
        
        # Preparar datos para graficar
        constraint_lines = []
        # Build labels depending on original input (string or structured)
        for orig, (coeffs, b, op) in zip(constraint_inputs, constraints):
            if isinstance(orig, str):
                label = orig.strip()
            elif isinstance(orig, dict):
                # try to build a human label
                label = orig.get('label') or f"{coeffs[0]}*x1 + {coeffs[1]}*x2 {op} {b}"
            else:
                label = f"{coeffs[0]}*x1 + {coeffs[1]}*x2 {op} {b}"

            constraint_lines.append({
                'coeffs': coeffs,
                'b': b,
                'op': op,
                'label': label
            })
        
        return jsonify({
            'success': True,
            'vertices': vertices,
            'optimal_point': optimal_point,
            'optimal_points': optimal_points,
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


@app.route('/favicon.ico')
def favicon():
    """Serve favicon if present in static, otherwise return 204 so browser doesn't log a 404."""
    ico_path = os.path.join(app.root_path, 'static', 'favicon.ico')
    if os.path.exists(ico_path):
        return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')
    return Response(status=204)
