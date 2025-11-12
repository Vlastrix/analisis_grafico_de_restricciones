// Funciones para manejar las restricciones
function addConstraint() {
    const container = document.getElementById('constraints-container');
    const newRow = document.createElement('div');
    newRow.className = 'constraint-row';
    newRow.innerHTML = `
        <input type="text" class="constraint-input" placeholder="Ej: x1 + x2 <= 5">
        <button class="btn-remove" onclick="removeConstraint(this)"><i class="fas fa-times"></i></button>
    `;
    container.appendChild(newRow);
}

function removeConstraint(button) {
    const container = document.getElementById('constraints-container');
    const rows = container.getElementsByClassName('constraint-row');
    
    // Mantener al menos una restricción
    if (rows.length > 1) {
        button.parentElement.remove();
    } else {
        alert('Debe haber al menos una restricción');
    }
}

// Cargar ejemplos predefinidos
function loadExample(exampleNum) {
    // Limpiar restricciones existentes
    const container = document.getElementById('constraints-container');
    container.innerHTML = '';
    
    if (exampleNum === 1) {
        // Ejemplo Clásico
        document.getElementById('obj_x1').value = 3;
        document.getElementById('obj_x2').value = 2;
        document.querySelector('input[name="objective_type"][value="maximize"]').checked = true;
        
        const constraints = [
            '2*x1 + x2 <= 10',
            'x1 + 2*x2 <= 8',
            'x1 <= 4'
        ];
        
        constraints.forEach(c => {
            const row = document.createElement('div');
            row.className = 'constraint-row';
            row.innerHTML = `
                <input type="text" class="constraint-input" value="${c}">
                <button class="btn-remove" onclick="removeConstraint(this)"><i class="fas fa-times"></i></button>
            `;
            container.appendChild(row);
        });
        
    } else if (exampleNum === 2) {
        // Ejemplo Problema de Producción
        document.getElementById('obj_x1').value = 40;
        document.getElementById('obj_x2').value = 30;
        document.querySelector('input[name="objective_type"][value="maximize"]').checked = true;
        
        const constraints = [
            'x1 + 2*x2 <= 12',
            '2*x1 + x2 <= 16',
            'x1 <= 7',
            'x2 <= 5'
        ];
        
        constraints.forEach(c => {
            const row = document.createElement('div');
            row.className = 'constraint-row';
            row.innerHTML = `
                <input type="text" class="constraint-input" value="${c}">
                <button class="btn-remove" onclick="removeConstraint(this)"><i class="fas fa-times"></i></button>
            `;
            container.appendChild(row);
        });
        
    } else if (exampleNum === 3) {
        // Ejemplo Minimización
        document.getElementById('obj_x1').value = 2;
        document.getElementById('obj_x2').value = 3;
        document.querySelector('input[name="objective_type"][value="minimize"]').checked = true;
        
        const constraints = [
            'x1 + x2 >= 4',
            '2*x1 + x2 >= 6',
            'x1 <= 5',
            'x2 <= 5'
        ];
        
        constraints.forEach(c => {
            const row = document.createElement('div');
            row.className = 'constraint-row';
            row.innerHTML = `
                <input type="text" class="constraint-input" value="${c}">
                <button class="btn-remove" onclick="removeConstraint(this)"><i class="fas fa-times"></i></button>
            `;
            container.appendChild(row);
        });
    }
}

// Resolver el problema
async function solveProblem() {
    // Ocultar resultados previos
    document.getElementById('results-panel').style.display = 'none';
    document.getElementById('error-panel').style.display = 'none';
    document.getElementById('loading').style.display = 'flex';
    document.getElementById('plot').style.display = 'none';
    
    // Recolectar datos del formulario
    const objX1 = parseFloat(document.getElementById('obj_x1').value) || 0;
    const objX2 = parseFloat(document.getElementById('obj_x2').value) || 0;
    const objectiveType = document.querySelector('input[name="objective_type"]:checked').value;
    
    const constraintInputs = document.getElementsByClassName('constraint-input');
    const constraints = [];
    for (let input of constraintInputs) {
        if (input.value.trim()) {
            constraints.push(input.value.trim());
        }
    }
    
    const x1Max = parseFloat(document.getElementById('x1_max').value) || 20;
    const x2Max = parseFloat(document.getElementById('x2_max').value) || 20;
    
    // Validaciones básicas
    if (constraints.length === 0) {
        showError('Debe ingresar al menos una restricción');
        return;
    }
    
    if (objX1 === 0 && objX2 === 0) {
        showError('La función objetivo no puede ser cero');
        return;
    }
    
    // Enviar solicitud al servidor
    try {
        const response = await fetch('/solve', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                obj_x1: objX1,
                obj_x2: objX2,
                objective_type: objectiveType,
                constraints: constraints,
                x1_max: x1Max,
                x2_max: x2Max
            })
        });
        
        const data = await response.json();
        
        document.getElementById('loading').style.display = 'none';
        
        if (data.success) {
            plotResults(data, x1Max, x2Max);
            showResults(data);
        } else {
            showError(data.error);
        }
        
    } catch (error) {
        document.getElementById('loading').style.display = 'none';
        showError('Error de conexión: ' + error.message);
    }
}

// Mostrar error
function showError(message) {
    document.getElementById('error-panel').style.display = 'block';
    document.getElementById('error-content').textContent = message;
    document.getElementById('plot').style.display = 'none';
}

// Mostrar resultados
function showResults(data) {
    const resultsPanel = document.getElementById('results-panel');
    const resultsContent = document.getElementById('results-content');
    
    if (!data.optimal_point || !data.optimal_value) {
        resultsContent.innerHTML = `
            <div class="result-item">
                <div class="result-label">Estado</div>
                <div class="result-value">No se encontró solución factible</div>
            </div>
        `;
        resultsPanel.style.display = 'block';
        return;
    }
    
    const [x1, x2] = data.optimal_point;
    const optimalValue = data.optimal_value;
    const objectiveType = data.is_maximize ? 'Maximizar' : 'Minimizar';
    
    resultsContent.innerHTML = `
        <div class="result-item">
            <div class="result-label">Tipo de Problema</div>
            <div class="result-value">${objectiveType}</div>
        </div>
        <div class="result-item">
            <div class="result-label">Punto Óptimo</div>
            <div class="result-value">x₁ = ${x1.toFixed(3)}, x₂ = ${x2.toFixed(3)}</div>
        </div>
        <div class="result-item">
            <div class="result-label">Valor Óptimo (Z)</div>
            <div class="result-value">${optimalValue.toFixed(3)}</div>
        </div>
        <div class="result-item">
            <div class="result-label">Número de Vértices</div>
            <div class="result-value">${data.vertices.length}</div>
        </div>
    `;
    
    resultsPanel.style.display = 'block';
}

// Graficar resultados con Plotly
function plotResults(data, x1Max, x2Max) {
    const traces = [];
    
    // Colores para las restricciones
    const colors = [
        '#ef4444', '#f59e0b', '#10b981', '#3b82f6', 
        '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'
    ];
    
    // 1. Graficar líneas de restricción
    data.constraint_lines.forEach((constraint, index) => {
        const [a1, a2] = constraint.coeffs;
        const b = constraint.b;
        
        const x1_vals = [];
        const x2_vals = [];
        
        // Calcular puntos para la línea
        if (Math.abs(a2) > 0.001) {
            // Línea no vertical
            for (let x1 = 0; x1 <= x1Max; x1 += 0.1) {
                const x2 = (b - a1 * x1) / a2;
                if (x2 >= 0 && x2 <= x2Max) {
                    x1_vals.push(x1);
                    x2_vals.push(x2);
                }
            }
        } else if (Math.abs(a1) > 0.001) {
            // Línea vertical
            const x1_val = b / a1;
            if (x1_val >= 0 && x1_val <= x1Max) {
                x1_vals.push(x1_val, x1_val);
                x2_vals.push(0, x2Max);
            }
        }
        
        if (x1_vals.length > 0) {
            traces.push({
                x: x1_vals,
                y: x2_vals,
                mode: 'lines',
                name: constraint.label,
                line: {
                    color: colors[index % colors.length],
                    width: 2
                },
                hovertemplate: `${constraint.label}<br>x₁: %{x:.2f}<br>x₂: %{y:.2f}<extra></extra>`
            });
        }
    });
    
    // 2. Graficar región factible
    if (data.vertices && data.vertices.length > 0) {
        const vertices = data.vertices;
        const x1_vertices = vertices.map(v => v[0]);
        const x2_vertices = vertices.map(v => v[1]);
        
        // Cerrar el polígono
        x1_vertices.push(vertices[0][0]);
        x2_vertices.push(vertices[0][1]);
        
        traces.push({
            x: x1_vertices,
            y: x2_vertices,
            fill: 'toself',
            fillcolor: 'rgba(16, 185, 129, 0.3)',
            line: {
                color: 'rgba(16, 185, 129, 0.8)',
                width: 2,
                dash: 'dot'
            },
            mode: 'lines',
            name: 'Región Factible',
            hoverinfo: 'skip'
        });
        
        // Marcar vértices
        traces.push({
            x: vertices.map(v => v[0]),
            y: vertices.map(v => v[1]),
            mode: 'markers',
            name: 'Vértices',
            marker: {
                color: '#10b981',
                size: 8,
                symbol: 'circle',
                line: {
                    color: 'white',
                    width: 2
                }
            },
            hovertemplate: 'Vértice<br>x₁: %{x:.3f}<br>x₂: %{y:.3f}<extra></extra>'
        });
    }
    
    // 3. Graficar línea de función objetivo
    if (data.optimal_point) {
        const [c1, c2] = data.objective_coeffs;
        const optimalValue = data.optimal_value;
        
        const x1_obj = [];
        const x2_obj = [];
        
        if (Math.abs(c2) > 0.001) {
            for (let x1 = 0; x1 <= x1Max; x1 += 0.1) {
                const x2 = (optimalValue - c1 * x1) / c2;
                if (x2 >= -1 && x2 <= x2Max + 1) {
                    x1_obj.push(x1);
                    x2_obj.push(x2);
                }
            }
        } else if (Math.abs(c1) > 0.001) {
            const x1_val = optimalValue / c1;
            x1_obj.push(x1_val, x1_val);
            x2_obj.push(0, x2Max);
        }
        
        if (x1_obj.length > 0) {
            traces.push({
                x: x1_obj,
                y: x2_obj,
                mode: 'lines',
                name: `Z = ${optimalValue.toFixed(2)}`,
                line: {
                    color: '#dc2626',
                    width: 3,
                    dash: 'dash'
                },
                hovertemplate: `Función Objetivo<br>x₁: %{x:.2f}<br>x₂: %{y:.2f}<extra></extra>`
            });
        }
    }
    
    // 4. Marcar punto óptimo
    if (data.optimal_point) {
        traces.push({
            x: [data.optimal_point[0]],
            y: [data.optimal_point[1]],
            mode: 'markers+text',
            name: 'Solución Óptima',
            marker: {
                color: '#dc2626',
                size: 20,
                symbol: 'star',
                line: {
                    color: 'white',
                    width: 2
                }
            },
            text: ['★ ÓPTIMO'],
            textposition: 'top center',
            textfont: {
                size: 12,
                color: '#dc2626',
                family: 'Arial Black'
            },
            hovertemplate: `<b>Punto Óptimo</b><br>x₁: ${data.optimal_point[0].toFixed(3)}<br>x₂: ${data.optimal_point[1].toFixed(3)}<br>Z: ${data.optimal_value.toFixed(3)}<extra></extra>`
        });
    }
    
    // Layout del gráfico
    const layout = {
        title: {
            text: 'Análisis Gráfico de Programación Lineal',
            font: {
                size: 20,
                color: '#1e293b'
            }
        },
        xaxis: {
            title: 'x₁',
            range: [0, x1Max],
            gridcolor: '#e2e8f0',
            showgrid: true,
            zeroline: true,
            zerolinecolor: '#94a3b8',
            zerolinewidth: 2
        },
        yaxis: {
            title: 'x₂',
            range: [0, x2Max],
            gridcolor: '#e2e8f0',
            showgrid: true,
            zeroline: true,
            zerolinecolor: '#94a3b8',
            zerolinewidth: 2
        },
        hovermode: 'closest',
        showlegend: true,
        legend: {
            x: 1.05,
            y: 1,
            bgcolor: 'rgba(255, 255, 255, 0.9)',
            bordercolor: '#e2e8f0',
            borderwidth: 1
        },
        plot_bgcolor: '#f8fafc',
        paper_bgcolor: 'white',
        margin: {
            l: 60,
            r: 150,
            t: 80,
            b: 60
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };
    
    document.getElementById('plot').style.display = 'block';
    Plotly.newPlot('plot', traces, layout, config);
}

// Inicializar con ejemplo al cargar la página
window.addEventListener('DOMContentLoaded', () => {
    console.log('Aplicación de Programación Lineal cargada');
});
