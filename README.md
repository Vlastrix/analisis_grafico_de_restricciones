# Análisis Gráfico de Restricciones para la Toma de Decisiones

Aplicación web educativa e interactiva para resolver y visualizar problemas de **Programación Lineal (PL)** con dos variables ($x_1$ y $x_2$).

## Características

- Interfaz intuitiva para ingresar restricciones y función objetivo
- Visualización gráfica interactiva de restricciones
- Identificación automática de la región factible
- Cálculo y resaltado del punto óptimo
- Soporte para problemas de maximización y minimización
- Gráficos interactivos con Plotly.js
- Herramienta educativa ideal para estudiantes de Investigación de Operaciones
- **Exportación completa a Word, PDF y Excel con pasos de resolución detallados**

## Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar o descargar el proyecto**

2. **Crear un entorno virtual (recomendado)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Instalar dependencias**
```powershell
pip install -r requirements.txt
```

4. **Ejecutar la aplicación**
```powershell
python app.py
```

5. **Abrir en el navegador**
```
http://localhost:5000
```

## Uso

### Formato de Restricciones

Las restricciones deben ingresarse en el siguiente formato:

- **Variables**: Usa `x1` o `X1` para $x_1$ y `x2` o `X2` para $x_2$
- **Operadores**: `<=`, `>=`, o `=`
- **Ejemplos válidos**:
  - `2*x1 + 3*x2 <= 12`
  - `x1 + x2 >= 4`
  - `3*x1 - x2 <= 6`
  - `x1 <= 5`
  - `x2 <= 4`

### Función Objetivo

Ingresa los coeficientes de la función objetivo:
- **Coeficiente de x₁**: Número que multiplica a $x_1$
- **Coeficiente de x₂**: Número que multiplica a $x_2$
- **Tipo**: Selecciona Maximizar o Minimizar

### Ejemplo Completo

**Problema:**
Maximizar: $Z = 3x_1 + 2x_2$

Sujeto a:
- $2x_1 + x_2 \leq 10$
- $x_1 + 2x_2 \leq 8$
- $x_1, x_2 \geq 0$

**Cómo ingresarlo:**
1. Función Objetivo:
   - Coeficiente x₁: `3`
   - Coeficiente x₂: `2`
   - Tipo: `Maximizar`

2. Restricciones:
   - `2*x1 + x2 <= 10`
   - `x1 + 2*x2 <= 8`

3. Hacer clic en "Resolver y Graficar"

4. **Exportar resultados** (nueva funcionalidad):
   - Una vez resuelto el problema, aparecerán botones de exportación
   - Haz clic en el formato deseado (Word, PDF o Excel)
   - Se descargará un documento completo con toda la solución paso a paso

## 📤 Exportación de Resultados

### ¿Qué incluye cada exportación?

Todos los formatos incluyen los siguientes pasos detallados:

1. **Paso 1: Definición del Problema**
   - Función objetivo completa
   - Tipo de optimización (maximizar/minimizar)

2. **Paso 2: Restricciones del Problema**
   - Lista completa de todas las restricciones
   - Restricciones de no negatividad

3. **Paso 3: Identificación de Vértices**
   - Coordenadas exactas de todos los vértices de la región factible

4. **Paso 4: Evaluación de la Función Objetivo**
   - Valor de Z en cada uno de los vértices encontrados
   - Cálculo detallado para cada punto

5. **Paso 5: Solución Óptima**
   - Punto óptimo (coordenadas exactas)
   - Valor óptimo de la función objetivo

6. **Visualización Gráfica** (Word y PDF)
   - Gráfico completo de la región factible
   - Restricciones, vértices y punto óptimo marcados

### Formatos Disponibles

#### Word (.docx)
- Documento profesional con formato estructurado
- Incluye el gráfico en alta calidad
- Ideal para reportes y trabajos académicos
- Fácilmente editable

#### PDF
- Reporte imprimible con diseño optimizado
- Perfecto para presentaciones
- Formato universal compatible con todos los dispositivos
- No requiere software especial para visualizar

#### Excel (.xlsx)
- Hoja de cálculo con datos estructurados
- Tabla detallada de vértices con evaluaciones
- Formato ideal para análisis adicionales
- Fácil de importar a otros programas

### Cómo Exportar

1. Resuelve tu problema de programación lineal
2. Revisa los resultados en pantalla
3. Desplázate al panel de resultados
4. Haz clic en el botón del formato deseado:
   - Exportar a Word
   - Exportar a PDF
   - Exportar a Excel
5. El archivo se descargará automáticamente

## Tecnologías Utilizadas

- **Backend**: Flask (Python)
- **Optimización**: SciPy (linprog)
- **Cálculos**: NumPy
- **Visualización**: Plotly.js
- **Frontend**: HTML5, CSS3, JavaScript
- **Exportación**:
  - python-docx (Word)
  - reportlab (PDF)
  - openpyxl (Excel)
  - plotly + kaleido (generación de imágenes)

## Componentes de la Visualización

1. **Líneas de Restricción**: Cada restricción se muestra como una línea de color diferente
2. **Región Factible**: Área sombreada en verde que cumple todas las restricciones
3. **Línea de Función Objetivo**: Línea roja punteada que representa valores constantes de la función objetivo
4. **Punto Óptimo**: Marcador rojo grande que indica la solución óptima
5. **Información Detallada**: Panel lateral con valores exactos de la solución
6. **Botones de Exportación**: Acceso rápido a exportar en múltiples formatos

## Aplicaciones Educativas

Esta herramienta es ideal para:
- Estudiantes de Investigación de Operaciones
- Cursos de Optimización
- Aprendizaje visual de conceptos de PL
- Verificación rápida de problemas de tarea
- Demostración en clase de métodos gráficos
- **Generación de reportes de laboratorio**
- **Documentación de soluciones para entregas académicas**

## Limitaciones

- Solo soporta problemas con **2 variables** (para visualización 2D)
- Las restricciones deben ser lineales
- Región factible debe ser acotada para mejor visualización

## Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto es de código abierto y está disponible para uso educativo.

## Autores

Vladislav Bochkov y Ender Rosales Condoli

## Reporte de Bugs

Si encuentras algún error, por favor abre un issue en el repositorio.

